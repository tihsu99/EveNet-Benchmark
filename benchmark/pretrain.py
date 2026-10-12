"""Optional small-pretraining stage using the same fixed-data preparation."""

import argparse
import json
import os
from pathlib import Path
import sys

from .common import (digest, file_hash, load_suite, merge, path_value, read_yaml,
                     resolve_template, run_command, source_revision, write_json)
from .prepare import verify_manifest
from .run import save_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("--prepared", required=True, help="Directory containing pretraining/manifest.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    suite, base = load_suite(args.suite)
    task = suite["pretraining"]
    core = path_value(suite["core"], base)
    prepared = Path(args.prepared).resolve() / "pretraining"
    manifest = verify_manifest(prepared)
    if manifest["selection"] != task["data"] or manifest["seed"] != suite["seed"]:
        raise ValueError("Prepared pretraining dataset does not match suite")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config = resolve_template(path_value(task["train_yaml"], base), core)
    config["options"] = merge(read_yaml(core / "share/options/options.yaml"), config.get("options", {}))
    config = merge(config, task.get("overrides", {}))
    config["platform"].update(data_parquet_dir=str(prepared / "train"),
                              data_parquet_val_dir=str(prepared / "validation"))
    config["options"]["Dataset"].update(dataset_limit=1.0, dataset_limit_apply_to_val_set=False,
        file_shuffle_seed=suite["seed"],
        normalization_file=str(prepared / "train" / task["normalization_asset"]))
    training = config["options"]["Training"]
    epochs = int(training["total_epochs"])
    training.update(epochs=epochs, model_checkpoint_save_top_k=1, random_seed=suite["seed"],
                    model_checkpoint_load_path=None, pretrain_model_load_path=None,
                    model_checkpoint_save_path=str(output / "checkpoints"))
    training["EarlyStopping"] = {"monitor": "val/loss", "mode": "min", "patience": epochs + 1,
                                 "min_delta": 0.0, "verbose": True}
    config["logger"]["local"] = {"save_dir": str(output / "logs"), "name": "pretraining", "version": "0"}
    config["logger"]["wandb"].update(project=suite.get("wandb_project", "EveNet-Benchmark"), run_name=output.name)
    config["logger"]["wandb"].pop("id", None)
    save_yaml(output / "train.yaml", config)
    runtime = suite.get("execution", {}).get("python", [sys.executable])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(core), str(Path(__file__).resolve().parent.parent),
                                                 environment.get("PYTHONPATH", "")])
    environment["PYTHONHASHSEED"] = str(suite["seed"])
    environment["WANDB_MODE"] = suite.get("execution", {}).get("wandb_mode", "offline")
    report = {"status": "running", "dataset_id": manifest["dataset_id"],
              "config": config, "core": source_revision(core), "core_submodule": source_revision(core / "evenet")}
    report["protocol_id"] = digest(report)
    write_json(output / "pretraining.json", report)
    try:
        run_command([*runtime, "-m", "benchmark.worker", "--seed", str(suite["seed"]), "script",
                     "--", str(core / "scripts/train.py"), str(output / "train.yaml"),
                     "--ray_dir", str(output / "ray")], output, output / "train.log", environment)
        run_command([*runtime, "-m", "benchmark.worker", "best", "--checkpoint", str(output / "checkpoints"),
                     "--output", str(output / "best.json")], output, output / "select.log", environment)
        best = json.loads((output / "best.json").read_text())
        report.update(status="complete", checkpoint={**best, "sha256": file_hash(best["path"])})
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    write_json(output / "pretraining.json", report)
    if report["status"] != "complete":
        raise SystemExit(1)
    print(best["path"])


if __name__ == "__main__":
    main()
