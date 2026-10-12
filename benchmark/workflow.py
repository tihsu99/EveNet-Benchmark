"""Download, launch, monitor and compare reproducible checkpoint probes."""

import argparse
from contextlib import contextmanager
import fcntl
import fnmatch
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

from tqdm.auto import tqdm

from .common import digest, load_suite, merge, path_value, read_yaml, run_command, write_json
from .scheduling import configured_platform, task_resources


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def expand(value, environment):
    if isinstance(value, dict):
        return {key: expand(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, environment) for item in value]
    if isinstance(value, str):
        return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}",
                      lambda match: environment.get(match[1], match[0]), value)
    return value


def configuration(path):
    path = Path(path).resolve()
    cfg = read_yaml(path)
    if cfg.get("schema_version") != 1:
        raise ValueError("Workflow needs schema_version: 1")
    root = path_value(cfg.get("download_root", cfg.get("workdir")), path.parent)
    environment = dict(os.environ)
    for key, item in {**cfg.get("sources", {}), **cfg.get("downloads", {})}.items():
        base = root / "data" if key in cfg.get("downloads", {}) else root
        environment[key] = str(path_value(item.get("path", item.get("local_dir")), base))
    for key, value in cfg.get("variables", {}).items():
        environment[key] = expand(value, environment)
    return cfg, path, root, environment


def fetch(cfg, root, environment, dry_run=False):
    """Resolve each Hub revision before downloading; preserve a source lock."""
    from huggingface_hub import HfApi, snapshot_download

    entries = {**cfg.get("sources", {}), **cfg.get("downloads", {})}
    receipt = {"schema_version": 1, "spec_id": digest(entries), "assets": {}}
    lock_path = root / "assets.lock.json"
    with locked(root / ".download.lock"):
        old = json.loads(lock_path.read_text()) if lock_path.exists() else None
        if old:
            # Compare only locked entries so additions also work with existing locks.
            previous_entries = {key: entries[key] for key in old["assets"] if key in entries}
            if (set(previous_entries) != set(old["assets"])
                    or digest(previous_entries) != old["spec_id"]):
                raise ValueError("Existing download/source specification changed; choose a new download_root")
        if old and any(Path(asset["path"]).resolve() != Path(environment[key]).resolve()
                       for key, asset in old["assets"].items()):
            raise ValueError("Download/source storage path changed; choose a new download_root")
        for key, source in tqdm(entries.items(), desc="Sources and downloads", unit="source"):
            target = Path(environment[key])
            if key in cfg.get("sources", {}):
                if source.get("path"):
                    if not target.is_dir():
                        raise FileNotFoundError(target)
                elif dry_run:
                    tqdm.write(f"Git {source['url']} @ {source['revision']} -> {target}")
                    continue
                elif not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".partial-" + uuid.uuid4().hex[:8])
                    try:
                        subprocess.run(["git", "init", str(temporary)], check=True)
                        subprocess.run(["git", "-C", str(temporary), "remote", "add", "origin", source["url"]], check=True)
                        subprocess.run(["git", "-C", str(temporary), "fetch", "--depth", "1", "origin", source["revision"]], check=True)
                        subprocess.run(["git", "-C", str(temporary), "checkout", "--detach", "FETCH_HEAD"], check=True)
                        subprocess.run(["git", "-C", str(temporary), "submodule", "update", "--init", "--recursive"], check=True)
                        temporary.rename(target)
                    except BaseException:
                        shutil.rmtree(temporary, ignore_errors=True)
                        raise
                commit = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
                if not source.get("path") and commit != source["revision"]:
                    raise ValueError(f"Source revision changed: {target}")
                receipt["assets"][key] = {"path": str(target), "revision": commit}
            else:
                revision = old["assets"][key]["revision"] if old and key in old["assets"] else source["revision"]
                info = HfApi().repo_info(source["repo_id"], repo_type=source.get("repo_type", "dataset"), revision=revision)
                patterns = source.get("allow_patterns", ["*"])
                ignored = source.get("ignore_patterns", [])
                files = [f.rfilename for f in info.siblings
                         if any(fnmatch.fnmatchcase(f.rfilename, p) for p in patterns)
                         and not any(fnmatch.fnmatchcase(f.rfilename, p) for p in ignored)]
                if not files:
                    raise ValueError(f"No Hub files selected for {key}")
                arguments = dict(repo_id=source["repo_id"], repo_type=source.get("repo_type", "dataset"),
                                 revision=info.sha, allow_patterns=files, local_dir=str(target))
                if dry_run:
                    planned = snapshot_download(**arguments, dry_run=True)
                    tqdm.write(f"{key}: {len(files)} files; {sum(f.file_size for f in planned):,} bytes selected; revision {info.sha} -> {target}")
                    continue
                tqdm.write(f"{key}: downloading {len(files)} files at revision {info.sha} -> {target}")
                snapshot_download(**arguments)
                if any(not (target / file).is_file() for file in files):
                    raise FileNotFoundError(f"Incomplete download: {key}")
                receipt["assets"][key] = {"path": str(target), "repo_id": source["repo_id"],
                                          "revision": info.sha, "files": files}
        if not dry_run:
            write_json(lock_path, receipt)
    return receipt


def resources(cfg):
    resource = cfg.get("resources", {})
    nodes = int(resource.get("nodes", 1))
    gpus = int(resource.get("gpus_per_node", 1))
    cpus = int(resource.get("cpus_per_node", 8))
    workers = int(resource.get("workers", nodes * max(gpus, 1)))
    cpus_per_worker = int(resource.get("cpus_per_worker", max(1, (cpus - 2) // max(gpus, 1))))
    max_parallel = int(resource.get("max_parallel", 1))
    if min(nodes, cpus, workers, cpus_per_worker, max_parallel) < 1 or gpus < 0:
        raise ValueError("Invalid resource count")
    if gpus and workers > nodes * gpus:
        raise ValueError("Workers exceed allocated GPUs")
    if workers * cpus_per_worker > nodes * cpus - 1:
        raise ValueError("Worker CPUs leave no capacity for the Ray driver")
    return {**resource, "nodes": nodes, "gpus_per_node": gpus, "cpus_per_node": cpus,
            "workers": workers, "cpus_per_worker": cpus_per_worker, "max_parallel": max_parallel}


def resolve_suite(cfg, path, environment):
    """Resolve the same task settings for preflight and actual execution."""
    suite, _ = load_suite(path_value(cfg["suite"], path.parent))
    suite = expand(suite, environment)
    conditions = cfg.get("conditions", {})
    selected = conditions.get("benchmarks")
    if selected is not None:
        unknown = set(selected) - suite["benchmarks"].keys()
        if unknown:
            raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")
        for name, task in suite["benchmarks"].items():
            task["enabled"] = name in selected
            if not task["enabled"]:
                task["reason"] = "Not selected by workflow conditions"
    suite = merge(suite, expand(conditions.get("suite_overrides", {}), environment))
    if not any(t.get("enabled", True) for t in suite["benchmarks"].values()):
        raise ValueError("Select at least one benchmark")
    resource = resources(cfg)
    platform = {"number_of_workers": resource["workers"], "use_gpu": resource["gpus_per_node"] > 0,
                "resources_per_worker": {"CPU": resource["cpus_per_worker"], "GPU": int(resource["gpus_per_node"] > 0)}}
    suite["adaptation"] = merge(suite.get("adaptation", {}), {"platform": platform})
    runtime = cfg.get("runtime", {})
    suite["execution"] = {**suite.get("execution", {}), "python": runtime.get("python", ["python3"]),
                          "analysis_python": runtime.get("analysis_python", runtime.get("python", ["python3"])),
                          "parallel": {"max_jobs": resource["max_parallel"],
                                       "CPU": resource["nodes"] * resource["cpus_per_node"] - resource.get("driver_cpus", 2),
                                       "GPU": resource["nodes"] * resource["gpus_per_node"]}}
    return suite, resource


def make_plan(cfg, path, root, environment, args):
    if not args.name or not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError("Use an alphanumeric checkpoint name, with underscores or hyphens")
    suite, resource = resolve_suite(cfg, path, environment)
    runtime = cfg.get("runtime", {})
    demands = {name: task_resources(configured_platform(suite, task, path.parent))
               for name, task in suite["benchmarks"].items() if task.get("enabled", True)}
    for name, demand in demands.items():
        if any(demand[key] > suite["execution"]["parallel"][key] for key in ("CPU", "GPU")):
            raise ValueError(f"Probe {name} needs {demand}; increase the allocation or explicitly change its platform")
        tqdm.write(f"{name}: {demand['GPU']} GPUs, {demand['CPU']} CPUs including trainer, "
                   f"batch {demand['batch_per_worker']} per worker / {demand['global_batch']} global")
    candidate = expand(cfg.get("candidate", {}), environment)
    checkpoint = None if args.scratch else str(path_value(args.checkpoint or candidate["checkpoint"], path.parent))
    network = str(path_value(args.network or candidate["network"], path.parent))
    version_dir = root / "version" / args.name
    if version_dir.exists():
        raise FileExistsError(f"Use a new checkpoint name: {version_dir}")
    plan_dir = version_dir / "job"
    plan_dir.mkdir(parents=True)
    suite_path = plan_dir / "suite.yaml"
    from .run import save_yaml
    save_yaml(suite_path, suite)
    load_suite(suite_path)  # Validate shared data selections after overrides.
    lock_path = root / "assets.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else None
    if lock is not None:
        # Later dataset additions must not invalidate already planned runs.
        lock_path = plan_dir / "assets.lock.json"
        write_json(lock_path, lock)
    plan = {"schema_version": 1, "name": args.name, "suite": str(suite_path), "network": network,
            "checkpoint": checkpoint, "weight_source": candidate.get("weight_source", "state_dict"),
            "prepared": str(root / "data" / "prepared"), "output": str(version_dir / "results"),
            "reference": str(path_value(cfg.get("reference", "paper_reference.yaml"), path.parent)),
            "asset_lock": str(lock_path), "asset_lock_id": digest(lock) if lock else None,
            "resources": resource, "probe_resources": demands,
            "runtime": runtime, "ray": expand(cfg.get("ray", {}), environment),
            "check_only": args.check_only, "created_at": time.time()}
    write_json(plan_dir / "plan.json", plan)
    return plan_dir / "plan.json"


def check_cluster(plan, output):
    """Attach to the user's cluster; never start, stop or resize Ray services."""
    configured = plan.get("ray", {}).get("address")
    address = (os.environ.get("RAY_ADDRESS") or configured) if configured in (None, "auto") else configured
    if not address:
        if plan["resources"]["nodes"] > 1 or plan["resources"].get("max_parallel", 1) > 1:
            raise ValueError("Multi-node or parallel probes require an existing Ray address")
        return
    if "$" in address:
        raise ValueError("Unresolved Ray address; set RAY_ADDRESS or ray.address")
    import ray
    os.environ["RAY_ADDRESS"] = address
    ray.init(address=address)
    try:
        request = plan["resources"]
        required = {"CPU": request["workers"] * request["cpus_per_worker"],
                    "GPU": request["workers"] if request["gpus_per_node"] else 0}
        if plan.get("probe_resources"):
            required = {key: max(d[key] for d in plan["probe_resources"].values()) for key in required}
        deadline = time.monotonic() + float(plan.get("ray", {}).get("wait_timeout", 300))
        while True:
            capacity, available = ray.cluster_resources(), ray.available_resources()
            nodes = sum(bool(node["Alive"]) for node in ray.nodes())
            write_json(output, {"address": address, "nodes": nodes, "capacity": capacity,
                                "available": available, "required": required})
            if nodes < request["nodes"] or any(capacity.get(k, 0) < v for k, v in required.items()):
                raise ValueError(f"Ray capacity too small: {nodes} nodes, {capacity}; required {required}")
            if all(available.get(k, 0) >= v for k, v in required.items()):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Ray resources remained busy; see {output}")
            tqdm.write(f"Waiting for Ray resources: available {available}, required {required}")
            time.sleep(5)
    finally:
        ray.shutdown()  # Disconnect this client only; the external cluster remains running.


def execute(plan_path):
    from .prepare import prepare_task, verify_manifest
    from .plot import render

    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    state_path = plan_path.parent / "status.json"
    state = {"status": "running", "stage": "ray-check", "started_at": time.time()}
    write_json(state_path, state)
    try:
        lock = json.loads(Path(plan["asset_lock"]).read_text())
        if digest(lock) != plan["asset_lock_id"]:
            raise ValueError("Asset lock changed since launch planning")
        check_cluster(plan, plan_path.parent / "ray.json")
        state["stage"] = "prepare"
        write_json(state_path, state)
        suite, base = load_suite(plan["suite"])
        if not Path(plan["network"]).is_file() or (plan["checkpoint"] and not Path(plan["checkpoint"]).is_file()):
            raise FileNotFoundError("Candidate checkpoint or network YAML missing; run download first")
        prepared = Path(plan["prepared"])
        groups = {t.get("data_group", n): t for n, t in suite["benchmarks"].items() if t.get("enabled", True)}
        with locked(prepared / ".prepare.lock"):
            for group, task in tqdm(groups.items(), desc="Fixed dataset preparation", unit="dataset"):
                target = prepared / group
                if target.exists():
                    manifest = verify_manifest(target)
                    if manifest["selection"] != task["data"] or manifest["seed"] != suite["seed"]:
                        raise ValueError(f"Prepared selection changed: {group}; use a new download_root")
                else:
                    temporary = prepared / (".partial-" + uuid.uuid4().hex)
                    try:
                        prepare_task(group, task, base, temporary, suite["seed"])
                        (temporary / group).rename(target)
                    finally:
                        shutil.rmtree(temporary, ignore_errors=True)
        state["stage"] = "benchmark"
        write_json(state_path, state)
        driver = plan.get("runtime", {}).get("driver", suite["execution"]["python"])
        argv = [*driver, "-m", "benchmark.run", plan["suite"],
                "--prepared", plan["prepared"], "--network", plan["network"],
                "--name", plan["name"], "--output", plan["output"], "--weight-source", plan["weight_source"],
                *(["--checkpoint", plan["checkpoint"]] if plan["checkpoint"] else ["--scratch"]),
                *(["--check-only"] if plan["check_only"] else [])]
        failure = None
        try:
            run_command(argv, plan_path.parent, plan_path.parent / "benchmark.log")
        except subprocess.CalledProcessError as error:
            failure = error
        report = Path(plan["output"]) / "report.json"
        if report.exists() and json.loads(report.read_text()).get("protocol_id"):
            state["stage"] = "report"
            write_json(state_path, state)
            render([report], Path(plan["output"]) / "figures", reference=plan["reference"])
        if failure:
            raise failure
        expected = "checked" if plan["check_only"] else "complete"
        if not report.exists() or json.loads(report.read_text()).get("status") != expected:
            raise RuntimeError(f"Benchmark did not produce a {expected} report: {report}")
        state.update(status="checked" if plan["check_only"] else "complete", stage="done", report=str(report))
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.time() - state["started_at"]
        write_json(state_path, state)


def status(plan_path, display=True):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    result = {"name": plan["name"], "status": "planned", "logs": str(plan_path.parent)}
    receipt = plan_path.parent / "submission.json"
    if receipt.exists() and not shutil.which("squeue"):
        result["scheduler_error"] = "squeue is unavailable here; reading shared status files only"
    elif receipt.exists():
        job = json.loads(receipt.read_text())["job_id"]
        result["job_id"] = job
        query = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T %R"], capture_output=True, text=True)
        if query.returncode:
            result["scheduler_error"] = query.stderr.strip()
        elif query.stdout.strip():
            result["scheduler"] = query.stdout.strip()
        elif shutil.which("sacct"):
            query = subprocess.run(["sacct", "-j", job, "--noheader", "--parsable2", "--format=JobID,State,ExitCode"], capture_output=True, text=True)
            result["scheduler"] = query.stdout.strip() or "Unavailable"
        else:
            result["scheduler_error"] = "Job absent from squeue; sacct unavailable"
    state = plan_path.parent / "status.json"
    if state.exists():
        result.update(json.loads(state.read_text()))
    result["stages"] = [json.loads(p.read_text()) for p in Path(plan["output"]).glob("*/progress.json")]
    suite, _ = load_suite(plan["suite"])
    result["total_probes"] = sum(t.get("enabled", True) for t in suite["benchmarks"].values())
    report = Path(plan["output"]) / "report.json"
    results = json.loads(report.read_text()).get("benchmarks", {}) if report.exists() else {}
    result["completed_probes"] = sum(r.get("status") in ("complete", "checked") for r in results.values())
    result["failed_probes"] = sum(r.get("status") == "failed" for r in results.values())
    result["running_probes"] = sum(r.get("status") == "running" for r in results.values())
    result["queued_probes"] = sum(r.get("status") == "queued" for r in results.values())
    if display:
        print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "run", "status", "compare", "_execute"])
    parser.add_argument("config", help="Workflow YAML; _execute/status also accept a plan JSON")
    parser.add_argument("--name")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--checkpoint")
    initialization.add_argument("--scratch", action="store_true")
    parser.add_argument("--network")
    parser.add_argument("--backend", choices=["local", "slurm"])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--reports", nargs="+")
    args = parser.parse_args()
    if args.action == "_execute":
        execute(args.config)
        return
    if args.action == "status":
        if Path(args.config).suffix == ".json":
            plan_path = Path(args.config)
        else:
            _, _, root, _ = configuration(args.config)
            if not args.name:
                parser.error("status requires --name")
            plan_path = root / "version" / args.name / "job" / "plan.json"
        if not args.watch:
            status(plan_path)
            return
        with tqdm(total=None, desc="Checkpoint", unit="probe") as progress:
            while True:
                result = status(plan_path, display=False)
                progress.total = result["total_probes"]
                progress.n = result["completed_probes"]
                progress.set_description(f"{result['name']}: {result.get('stage', result.get('scheduler', result['status']))}")
                active = next((s for s in result["stages"] if s["status"] == "running"), {})
                progress.set_postfix(stage=active.get("stage", "-"), running=result["running_probes"],
                                     queued=result["queued_probes"], failed=result["failed_probes"])
                progress.refresh()
                if result["status"] in ("complete", "checked", "failed") or any(
                    word in result.get("scheduler", "") for word in ("FAILED", "CANCELLED", "TIMEOUT", "COMPLETED", "OUT_OF_MEMORY", "NODE_FAIL")):
                    break
                time.sleep(10)
        return
    cfg, path, root, environment = configuration(args.config)
    if args.action == "download":
        fetch(cfg, root, environment, args.dry_run)
    elif args.action == "compare":
        from .plot import render
        reports = args.reports or sorted((root / "version").glob("*/results/report.json"))
        if not reports:
            parser.error("No checkpoint reports found")
        print(render(reports, root / "comparison", reference=path_value(cfg.get("reference", "paper_reference.yaml"), path.parent)))
    else:
        if not args.dry_run and not (root / "assets.lock.json").exists():
            raise FileNotFoundError("Run 'evenet-benchmark download <workflow.yaml>' first")
        plan_path = make_plan(cfg, path, root, environment, args)
        backend = args.backend or cfg.get("backend", "local")
        if backend == "slurm":
            from .slurm import submit
            submit(plan_path, dry_run=args.dry_run)
        elif backend == "local":
            print(plan_path)
            if not args.dry_run:
                driver = cfg.get("runtime", {}).get("driver", [sys.executable])
                environment = dict(os.environ)
                environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH", "")])
                run_command([*driver, "-m", "benchmark.workflow", "_execute", str(plan_path)],
                            plan_path.parent, plan_path.parent / "driver.log", environment)
        else:
            raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    main()
