"""Adapt independent copies of one checkpoint and write a compact report JSON."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import yaml
from tqdm.auto import tqdm

from .common import (digest, file_hash, load_suite, merge, path_value, read_yaml,
                     resolve_template, run_command, source_revision, write_json)
from .prepare import verify_manifest
from .scheduling import run_parallel, task_resources


def save_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False))


def save_model_config(path, config):
    """Keep EventInfo reloadable by the native driver and Ray workers."""
    path = Path(path)
    config = copy.deepcopy(config)
    if "event_info" in config:
        event_path = path.with_name(f"{path.stem}.event_info.yaml")
        save_yaml(event_path, config["event_info"])
        # The native default branch replaces the previously constructed EventInfo.
        config["event_info"] = {"default": event_path.name}
    save_yaml(path, config)


def build_configs(suite, task, base, prepared, work, network, checkpoint):
    core = path_value(suite["core"], base)
    repository = path_value(task["repository"], base)
    template = resolve_template(path_value(task["train_yaml"], base), repository, task.get("template_tokens"))
    train = copy.deepcopy(template)
    train["options"] = merge(read_yaml(core / "share/options/options.yaml"), train.get("options", {}))
    # Candidate architecture is explicit; task dimensions/features can be overridden in YAML.
    train["network"] = merge(network, task.get("network_overrides", {}))
    train = merge(train, suite.get("adaptation", {}))
    train = merge(train, task.get("overrides", {}))
    options = train["options"]
    options["Dataset"].update({"dataset_limit": 1.0, "dataset_limit_apply_to_val_set": False,
                               "file_shuffle_seed": suite["seed"],
                               "normalization_file": str(prepared / "train" / task["normalization_asset"])})
    options["Dataset"]["balance_file"] = None
    options["Metrics"].setdefault("PairContrastive", {})["enabled"] = False
    options["Metrics"].setdefault("PairRepresentation", {})["enabled"] = False
    training = options["Training"]
    # A downstream task must not inherit pretraining-only truth objectives.
    training["Components"].setdefault("PairContrastive", {})["include"] = False
    training.update({"random_seed": suite["seed"], "model_checkpoint_save_top_k": 1,
                     "model_checkpoint_save_path": str(work / "checkpoints"),
                     "model_checkpoint_load_path": None,
                     "pretrain_model_load_path": str(work / "initialization.ckpt") if checkpoint else None})
    epochs = int(training["total_epochs"])
    training["epochs"] = epochs
    training["EarlyStopping"] = {"monitor": "val/loss", "mode": "min", "min_delta": 0.0,
                                 "patience": epochs + 1, "verbose": True}
    # Raw adapted weights are evaluated consistently, including the native anomaly generator.
    training["EMA"]["replace_model_after_load"] = False
    training["EMA"]["replace_model_at_end"] = False
    train["platform"].update({"data_parquet_dir": str(prepared / "train"),
                               "data_parquet_val_dir": str(prepared / "validation")})
    logger = train.setdefault("logger", {})
    logger.setdefault("local", {}).update({"save_dir": str(work / "logs"), "name": "probe", "version": "adapt"})
    logger.setdefault("wandb", {}).update({"project": suite.get("wandb_project", "EveNet-Benchmark"),
                                           "run_name": work.parent.name + "-" + work.name})
    logger["wandb"].pop("id", None)
    logger["wandb"].pop("entity", None)
    prediction = copy.deepcopy(train)
    if task.get("predict_yaml"):
        template_predict = resolve_template(path_value(task["predict_yaml"], base), repository, task.get("template_tokens"))
        prediction["options"]["prediction"] = template_predict.get("options", {}).get("prediction", {})
    prediction["options"].setdefault("prediction", {}).update({
        "output_dir": str(work / "predictions"), "filename": "prediction.pt"})
    prediction["platform"]["data_parquet_dir"] = str(prepared / "evaluation")
    prediction["options"]["Dataset"]["dataset_limit"] = 1.0
    prediction["options"]["Training"]["pretrain_model_load_path"] = None
    return train, prediction


def merge_parquet(inputs, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted(Path(inputs).glob("*.parquet"))
    with pq.ParquetWriter(output, pq.ParquetFile(files[0]).schema_arrow) as writer:
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
                writer.write_table(pa.Table.from_batches([batch]))


def anomaly_workflow(task, base, prepared, work):
    repository = path_value(task["repository"], base)
    workflow_path = path_value(task["workflow_yaml"], base)
    workflow = read_yaml(workflow_path)
    for section in ("input",):
        workflow[section]["event_info"] = str(path_value(workflow[section]["event_info"], workflow_path.parent))
    workflow["train"]["config"] = str(work / "train.yaml")
    workflow["workdir"] = str(repository)
    workflow["output"] = {"storedir": str(work / "data"), "plotdir": str(work / "figures")}
    for region, split in (("SB", "train"), ("SR", "evaluation")):
        folder = work / "data" / region
        folder.mkdir(parents=True)
        merge_parquet(prepared / split, folder / "data.parquet")
        for asset in task["data"]["assets"]:
            shutil.copyfile(prepared / split / asset, folder / asset)
    save_yaml(work / "workflow.yaml", workflow)
    return workflow


def run_task(name, task, suite, base, prepared, work, network, checkpoint, weight_source, check_only=False,
             cancel_event=None):
    core = path_value(suite["core"], base)
    repository = path_value(task["repository"], base)
    execution = suite.get("execution", {})
    runtime = execution.get("python", [sys.executable])
    analysis_runtime = execution.get("analysis_python", runtime)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(core), str(Path(__file__).resolve().parent.parent),
                                                 environment.get("PYTHONPATH", "")])
    environment["PYTHONHASHSEED"] = str(suite["seed"])
    environment["WANDB_MODE"] = execution.get("wandb_mode", "offline")
    work.mkdir(parents=True)
    train, prediction = build_configs(suite, task, base, prepared, work, network, checkpoint)
    save_model_config(work / "train.yaml", train)
    save_yaml(work / "evaluation.yaml", task["evaluation"])
    save_yaml(work / "load_policy.yaml", suite["load_policy"])
    commands = []

    def worker(operation, arguments, label, analysis=False, reserve_gpu=False):
        argv = [*(analysis_runtime if analysis else runtime), "-m", "benchmark.worker",
                "--seed", str(suite["seed"]), operation, *map(str, arguments)]
        if reserve_gpu:
            prefix = analysis_runtime if analysis else runtime
            argv = [*prefix, "-m", "benchmark.ray_script", "--cpus",
                    str(train["platform"]["resources_per_worker"]["CPU"]), "--gpus", "1",
                    "--", *argv[len(prefix):]]
        commands.append({"stage": label, "argv": argv, "cwd": str(work)})
        write_json(work / "commands.json", commands)
        total = 1 if check_only else 5
        state = {"stage": label, "stage_number": len(commands), "total_stages": total,
                 "status": "running", "started_at": time.time(), "log": str(work / f"{label}.log")}
        write_json(work / "progress.json", state)
        tqdm.write(f"[{name}] {len(commands)}/{total} {label} | log: {state['log']}")
        try:
            with tqdm(total=total, initial=len(commands)-1, desc=f"{name}: {label}", unit="stage",
                      disable=not sys.stderr.isatty(), leave=False) as progress:
                run_command(argv, work, work / f"{label}.log", environment,
                            **({"cancel_event": cancel_event} if cancel_event is not None else {}))
                progress.update(1)
            state["status"] = "complete"
        except BaseException:
            state["status"] = "failed"
            raise
        finally:
            state["elapsed_seconds"] = time.time() - state["started_at"]
            write_json(work / "progress.json", state)

    worker("audit", ["--config", work / "train.yaml", "--policy", work / "load_policy.yaml",
                     "--output", work / "load_audit.json", "--weight-source", weight_source,
                     *(["--checkpoint", checkpoint] if checkpoint else [])], "audit")
    if check_only:
        return {"status": "checked", "load_audit": json.loads((work / "load_audit.json").read_text()),
                "resolved_training_config": str(work / "train.yaml")}
    worker("script", ["--", core / "scripts/train.py", work / "train.yaml",
                      "--ray_dir", work / "ray"], "adapt")
    worker("best", ["--checkpoint", work / "checkpoints", "--output", work / "best.json"], "select")
    best = json.loads((work / "best.json").read_text())
    prediction["options"]["Training"]["model_checkpoint_load_path"] = best["path"]
    save_model_config(work / "predict.yaml", prediction)
    kind = task["evaluation"]["kind"]
    if kind == "anomaly":
        anomaly_workflow(task, base, prepared, work)
        evaluation = task["evaluation"]
        worker("script", ["--", repository / "01predict_SR_certain_fold.py", work / "workflow.yaml",
                          "--checkpoint", best["path"], "--region", "SR", "--fold", task["data"]["fold"]["index"],
                          "--gen_num_events", evaluation["generated_events"],
                          "--batch_size", evaluation.get("batch_size", train["platform"]["batch_size"]),
                          "--ngpu", "1", "--rank", "0"], "predict", reserve_gpu=True)
        worker("script", ["--", repository / "02prepare_classification_dataset.py", work / "workflow.yaml",
                          "--region", "SR", "--calibrated", "--max_background", evaluation["max_background"]],
               "evaluate", analysis=True, reserve_gpu=True)
        source = work / "data_hybrid_calibrated/SR/gen_performance.json"
        native = json.loads(source.read_text())
        metrics = {key: native[section][metric] for key, (section, metric) in evaluation["metric_paths"].items()}
        evaluation_result = {"metrics": metrics, "source": str(source),
                             "notes": ["One fixed fold; generation-quality probe only, not bump-hunt significance",
                                       "Native generator uses seed 42 on rank zero"]}
        write_json(work / "metrics.json", evaluation_result)
    else:
        worker("script", ["--", core / "scripts/predict.py", work / "predict.yaml"], "predict")
        worker(kind, ["--prediction", work / "predictions/prediction.pt", "--config", work / "evaluation.yaml",
                      "--repository", repository, "--output", work / "metrics.json"], "evaluate", analysis=True)
        evaluation_result = json.loads((work / "metrics.json").read_text())
    return {"status": "complete", "evaluation": evaluation_result,
            "adapted_checkpoint": {**best, "sha256": file_hash(best["path"])},
            "load_audit": json.loads((work / "load_audit.json").read_text()),
            "resolved_training_config": str(work / "train.yaml")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--network", required=True, help="Candidate network YAML or full pretraining YAML")
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--checkpoint")
    initialization.add_argument("--scratch", action="store_true")
    parser.add_argument("--weight-source", choices=["state_dict", "ema_state_dict"], default="state_dict")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-only", action="store_true", help="Verify data and CPU model transfer without training")
    args = parser.parse_args()
    suite, base = load_suite(args.suite)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    report = {"schema_version": 1, "name": args.name, "status": "running", "benchmarks": {},
              "suite": suite, "candidate": {"mode": "scratch" if args.scratch else "pretrained"}}
    report_path = output / "report.json"
    write_json(report_path, report)
    try:
        core = path_value(suite["core"], base)
        network = resolve_template(Path(args.network).resolve())
        network = network.get("network", network)
        original_checkpoint = str(Path(args.checkpoint).resolve()) if args.checkpoint else None
        checkpoint, checkpoint_hash = None, None
        if original_checkpoint:
            checkpoint_hash = file_hash(original_checkpoint)
            checkpoint = str(output / "candidate.ckpt")
            shutil.copyfile(original_checkpoint, checkpoint)
            if file_hash(checkpoint) != checkpoint_hash:
                raise ValueError("Candidate checkpoint changed while being snapshotted")
        report["candidate"].update({"checkpoint": checkpoint, "original_path": original_checkpoint,
                                     "sha256": checkpoint_hash,
                                     "network": network, "weight_source": args.weight_source})
        provenance = {"core": source_revision(core), "core_submodule": source_revision(core / "evenet"),
                      "benchmarks": {}, "datasets": {}, "recipes": {},
                      "runner_sources": {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")}}
        manifests, verified_groups = {}, {}
        for name, task in suite["benchmarks"].items():
            if not task.get("enabled", True):
                report["benchmarks"][name] = {"status": "skipped", "reason": task.get("reason", "Disabled in YAML")}
                continue
            group = task.get("data_group", name)
            prepared = Path(args.prepared).resolve() / group
            if group not in verified_groups:
                verified_groups[group] = verify_manifest(prepared)
            manifests[name] = verified_groups[group]
            if manifests[name]["selection"] != task["data"] or manifests[name]["seed"] != suite["seed"]:
                raise ValueError(f"Prepared dataset does not match suite: {name}")
            repository = path_value(task["repository"], base)
            if not repository.is_dir():
                raise FileNotFoundError(repository)
            provenance["benchmarks"][name] = source_revision(repository)
            provenance["datasets"][name] = manifests[name]["dataset_id"]
            provenance["recipes"][name] = {
                key: resolve_template(path_value(task[key], base), repository, task.get("template_tokens"))
                for key in ("train_yaml", "predict_yaml") if task.get(key)}
            if task.get("workflow_yaml"):
                provenance["recipes"][name]["workflow"] = read_yaml(path_value(task["workflow_yaml"], base))
        if not manifests:
            raise ValueError("At least one benchmark must be enabled")
        provenance["core_options"] = read_yaml(core / "share/options/options.yaml")
        report["provenance"] = provenance
        adaptation_protocol = {key: value for key, value in suite.items() if key != "pretraining"}
        report["protocol_id"] = digest({"suite": adaptation_protocol, "provenance": provenance})
        write_json(report_path, report)
        enabled = [(name, task) for name, task in suite["benchmarks"].items() if task.get("enabled", True)]
        demands = {name: task_resources(build_configs(
            suite, task, base, Path(args.prepared).resolve() / task.get("data_group", name),
            output / name, network, checkpoint)[0]["platform"]) for name, task in enabled}
        report["resources"] = demands
        for name, _ in enabled:
            report["benchmarks"][name] = {"status": "queued", "resources": demands[name]}
        write_json(report_path, report)

        def on_start(name):
            report["benchmarks"][name] = {"status": "running", "resources": demands[name]}
            write_json(report_path, report)

        cancellation = threading.Event()

        def run(name, task):
            start = time.monotonic()
            try:
                result = run_task(name, task, suite, base, Path(args.prepared).resolve() / task.get("data_group", name),
                                  output / name, network, checkpoint, args.weight_source, args.check_only,
                                  cancel_event=cancellation)
                if not args.check_only and task["evaluation"]["kind"] != "anomaly" and result["evaluation"].get("events") != manifests[name]["counts"]["evaluation"]:
                    raise ValueError("Prediction row count does not match the fixed evaluation manifest")
            except Exception as error:
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                          "logs": str(output / name)}
            result["elapsed_seconds"] = time.monotonic() - start
            result["counts"] = manifests[name]["counts"]
            result["resources"] = demands[name]
            return result

        parallel = suite.get("execution", {}).get("parallel", {})
        if int(parallel.get("max_jobs", 1)) > 1 and not os.environ.get("RAY_ADDRESS"):
            raise ValueError("Parallel probes require RAY_ADDRESS for an existing shared cluster")
        for name, future in run_parallel(enabled, run, demands, parallel, on_start, cancellation):
            report["benchmarks"][name] = future.result()
            write_json(report_path, report)
        successful = "checked" if args.check_only else "complete"
        report["status"] = successful if all(result["status"] in (successful, "skipped")
                                              for result in report["benchmarks"].values()) else "partial"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if not isinstance(error, Exception):
            raise
    finally:
        write_json(report_path, report)
    print(report_path)
    if report["status"] not in ("complete", "checked"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
