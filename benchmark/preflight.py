"""Check configured runtimes and input headers without training or preparing data."""

import argparse
from datetime import datetime, timezone
import glob
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

from .common import merge, path_value, resolve_template, write_json
from .run import build_configs
from .workflow import configuration, expand, resolve_suite


def record(checks, name, operation):
    try:
        detail = operation()
        result = {"name": name, "status": "pass", "detail": detail}
    except Exception as error:
        result = {"name": name, "status": "fail", "error": f"{type(error).__name__}: {error}"}
        traceback.print_exc()
    checks.append(result)
    print(f"[{result['status'].upper()}] {name}: {result.get('error', '')}", flush=True)
    return result


def probe(spec, output):
    """Run inside the target Python; native script main blocks are never called."""
    checks = []
    for name in spec.get("imports", []):
        def import_module(name=name):
            module = importlib.import_module(name)
            return {"version": str(getattr(module, "__version__", "unknown")),
                    "file": getattr(module, "__file__", None)}
        record(checks, f"import:{name}", import_module)
    for key in spec.get("required_env", []):
        def environment(key=key):
            if not os.environ.get(key):
                raise ValueError(f"Required environment variable is not set: {key}")
            return "present"
        record(checks, f"environment:{key}", environment)
    for script in spec.get("scripts", []):
        def import_script(script=script):
            sys.path.insert(0, str(Path(script).parent))
            try:
                runpy.run_path(script, run_name="_benchmark_preflight_import")
            finally:
                sys.path.pop(0)
            return str(script)
        record(checks, f"script-import:{script}", import_script)
    if "quantum" in spec:
        def quantum():
            from .worker import load_quantum_analysis
            load_quantum_analysis(**spec["quantum"])
            return "Native analysis imported; RooUnfoldResponse and RooUnfoldSvd available"
        record(checks, "quantum-native-analysis", quantum)
    if "ray_address" in spec:
        def cluster():
            import ray
            ray.init(address=spec["ray_address"])
            try:
                return {"nodes": sum(bool(node["Alive"]) for node in ray.nodes()),
                        "capacity": ray.cluster_resources(), "available": ray.available_resources()}
            finally:
                ray.shutdown()
        record(checks, "ray-existing-cluster", cluster)
    report = {"status": "pass" if all(c["status"] == "pass" for c in checks) else "fail",
              "python": sys.executable, "python_version": sys.version, "checks": checks}
    write_json(output, report)
    return report


def input_headers(task, base):
    import pyarrow.parquet as pq
    details = {}
    for split in ("train", "validation", "evaluation"):
        patterns = task["data"].get(split, [])
        patterns = [patterns] if isinstance(patterns, str) else patterns
        if not patterns:
            if split != "validation":
                raise ValueError(f"No {split} input configured")
            continue
        files = sorted({item for pattern in patterns
                        for item in glob.glob(str(path_value(pattern, base)), recursive=True)})
        if not files:
            raise FileNotFoundError(f"No {split} files: {patterns}")
        header = pq.ParquetFile(files[0])
        details[split] = {"files": len(files), "sample_file": files[0],
                          "sample_rows": header.metadata.num_rows}
    for asset in task["data"].get("assets", {}).values():
        if not path_value(asset, base).is_file():
            raise FileNotFoundError(asset)
    return details


def run(config, output, timeout=120, ray=False):
    cfg, path, root, environment = configuration(config)
    suite, _ = resolve_suite(cfg, path, environment)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks, runtimes = [], {}
    candidate = expand(cfg.get("candidate", {}), environment)

    def candidate_files():
        for key in ("network", "checkpoint"):
            if not path_value(candidate[key], path.parent).is_file():
                raise FileNotFoundError(candidate[key])
        return candidate

    record(checks, "candidate-files", candidate_files)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(path_value(suite["core"], path.parent)),
        str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = str(suite["seed"])

    def runtime(label, argv, spec, wandb_mode="offline"):
        process_env = {**env, "WANDB_MODE": wandb_mode}
        key = json.dumps([argv, spec, wandb_mode], sort_keys=True)
        if key in runtimes:
            checks.append({**runtimes[key], "name": label, "reused": runtimes[key]["name"]})
            print(f"[{runtimes[key]['status'].upper()}] {label} (shared runtime check)", flush=True)
            return
        folder = output / label
        folder.mkdir()
        write_json(folder / "spec.json", spec)
        command = [*argv, "-m", "benchmark.preflight", "--probe", str(folder / "spec.json"),
                   "--output", str(folder / "report.json")]

        def execute():
            with (folder / "probe.log").open("w") as log:
                log.write(json.dumps({"argv": command, "cwd": str(folder)}) + "\n")
                log.flush()
                result = subprocess.run(command, cwd=folder, env=process_env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout)
            if (folder / "report.json").is_file():
                report = json.loads((folder / "report.json").read_text())
                failures = [f"{item['name']}: {item['error']}" for item in report["checks"]
                            if item["status"] == "fail"]
                if failures:
                    raise RuntimeError("; ".join(failures))
            if result.returncode:
                raise RuntimeError(f"Exit {result.returncode}; inspect {folder / 'probe.log'}")
            report = json.loads((folder / "report.json").read_text())
            if report["status"] != "pass":
                raise RuntimeError(f"Runtime checks failed; inspect {folder / 'report.json'}")
            return {"python": report["python"], "python_version": report["python_version"]}

        result = record(checks, label, execute)
        result["log"] = str(folder / "probe.log")
        if (folder / "report.json").is_file():
            result["probe"] = json.loads((folder / "report.json").read_text())
        runtimes[key] = result

    for name, task in suite["benchmarks"].items():
        if not task.get("enabled", True):
            continue
        record(checks, f"{name}-inputs", lambda task=task: input_headers(task, path.parent))

        def model_config(task=task, name=name):
            network = resolve_template(path_value(candidate["network"], path.parent))
            train, _ = build_configs(suite, task, path.parent, root / "data/prepared" /
                task.get("data_group", name), output / name, network.get("network", network),
                candidate.get("checkpoint"))
            if train["options"]["Training"]["Components"].get("Segmentation", {}).get("include"):
                metadata = json.loads(path_value(task["data"]["assets"]["shape_metadata.json"],
                                                path.parent).read_text())
                targets = metadata["segmentation-data"][0]
                queries = train["network"]["Segmentation"]["num_queries"]
                if queries != targets:
                    raise ValueError(f"Segmentation queries={queries}, target slots={targets}; "
                                     "set task.network_overrides.Segmentation.num_queries")
            return "Resolved native templates and task overrides"

        record(checks, f"{name}-config", model_config)
        execution = merge(suite.get("execution", {}), task.get("execution", {}))
        training = execution.get("python", [sys.executable])
        analysis = execution.get("analysis_python", training)
        settings = merge(suite.get("preflight", {}), task.get("preflight", {}))
        for phase in ("training", "prediction", "evaluation"):
            spec = settings.get(phase)
            if spec is None:
                if phase != "prediction":
                    checks.append({"name": f"{name}-{phase}", "status": "fail",
                                   "error": f"No preflight.{phase} checks configured"})
                continue
            if not any(spec.get(key) for key in ("imports", "scripts", "required_env")):
                checks.append({"name": f"{name}-{phase}", "status": "fail",
                               "error": f"Empty preflight.{phase} checks"})
                continue
            spec = dict(spec)
            if phase == "evaluation" and task["evaluation"]["kind"] == "quantum":
                spec["quantum"] = {"repository": str(path_value(task["repository"], path.parent)),
                                   "config": task["evaluation"]}
            runtime(f"{name}-{phase}", analysis if phase == "evaluation" else training, spec,
                    execution.get("wandb_mode", "offline"))
    if ray:
        configured = cfg.get("ray", {}).get("address")
        address = os.environ.get("RAY_ADDRESS", configured) if configured in (None, "auto") else configured
        if not address:
            checks.append({"name": "ray", "status": "fail", "error": "No existing Ray address configured"})
        else:
            runtime("ray", cfg.get("runtime", {}).get("driver", [sys.executable]),
                    {"ray_address": address})
    report = {"status": "pass" if all(c["status"] == "pass" for c in checks) else "fail",
              "config": str(path), "checks": checks,
              "not_tested": ["checkpoint transfer", "full dataset integrity", "GPU kernels",
                             "distributed worker environments", "training", "physics evaluation"]}
    if not ray:
        report["not_tested"].append("Ray cluster connection")
    write_json(output / "report.json", report)
    print(f"Preflight {report['status'].upper()}: {output / 'report.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="Workflow YAML")
    parser.add_argument("--output", help="New directory for reports, or probe report path")
    parser.add_argument("--timeout", type=float, default=120, help="Timeout per runtime check, seconds")
    parser.add_argument("--ray", action="store_true", help="Also attach to an existing cluster without jobs")
    parser.add_argument("--probe", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        result = probe(json.loads(Path(args.probe).read_text()), args.output)
    else:
        if not args.config or args.timeout <= 0:
            parser.error("Supply a workflow YAML and a positive timeout")
        output = args.output or str(configuration(args.config)[2] / "preflight" /
                                    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        result = run(args.config, output, args.timeout, args.ray)
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
