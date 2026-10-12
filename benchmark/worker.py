"""EveNet-runtime operations, isolated from the lightweight runner."""

import argparse
from contextlib import contextmanager
from fnmatch import fnmatchcase
import importlib.util
import json
import os
from pathlib import Path
import random
import runpy
import sys

import numpy as np

from .common import read_yaml, write_json


@contextmanager
def ray_storage(directory):
    """Constrain native scripts' Ray results to the benchmark output directory."""
    if directory is None:
        yield
        return
    from ray.train import RunConfig

    directory = str(Path(directory).resolve())
    Path(directory).mkdir(parents=True, exist_ok=True)
    original = RunConfig.__post_init__

    def configure(config):
        # Native prediction scripts omit storage_path and otherwise use ~/ray_results.
        config.storage_path = directory
        original(config)

    RunConfig.__post_init__ = configure
    try:
        yield
    finally:
        RunConfig.__post_init__ = original


def audit_transfer(target, source, required, allowed_missing=(), allowed_unused=()):
    """Check both directions: critical source parameters cannot disappear silently."""
    matched = {key: value for key, value in source.items()
               if key in target and value.shape == target[key].shape}
    required_key = lambda key: any(fnmatchcase(key, pattern) for pattern in required)
    missing = [key for key in target if key not in matched and required_key(key)
               and not any(fnmatchcase(key, pattern) for pattern in allowed_missing)]
    unused = [key for key in source if key not in matched and required_key(key)
              and not any(fnmatchcase(key, pattern) for pattern in allowed_unused)]
    return matched, {"passed": not missing and not unused,
                     "required_missing": missing, "required_unused": unused,
                     "matched_numel": sum(value.numel() for value in matched.values()),
                     "matched_keys": sorted(matched),
                     "all_missing": sorted(set(target) - set(matched)),
                     "all_unused": sorted(set(source) - set(matched))}


def audit(args):
    import torch
    from evenet.control.global_config import global_config
    from evenet.network.evenet_model import EveNetModel
    global_config.load_yaml(args.config)
    components = global_config.options.Training.Components
    switches = {"classification": "Classification", "regression": "Regression",
                "assignment": "Assignment", "segmentation": "Segmentation",
                "global_generation": "GlobalGeneration", "point_cloud_generation": "ReconGeneration",
                "neutrino_generation": "TruthGeneration"}
    model = EveNetModel(config=global_config, device=torch.device("cpu"),
                       normalization_dict=torch.load(global_config.options.Dataset.normalization_file,
                                                     map_location="cpu", weights_only=False),
                       **{key: bool(components[value].get("include", False)) for key, value in switches.items()})
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if args.weight_source not in checkpoint:
            raise ValueError(f"Checkpoint lacks {args.weight_source}")
        source = {key.removeprefix("model."): value for key, value in checkpoint[args.weight_source].items()}
        # Match the core's intentional task-specific normalization replacement.
        source = {key: value for key, value in source.items() if "_normalizer" not in key}
        policy = read_yaml(args.policy)
        matched, report = audit_transfer(model.state_dict(), source, **policy)
        report["weight_source"] = args.weight_source
        write_json(args.output, report)
        if not report["passed"]:
            raise ValueError("Checkpoint transfer audit failed; inspect load_audit.json")
        torch.save({"state_dict": matched}, Path(args.output).with_name("initialization.ckpt"))
    else:
        write_json(args.output, {"passed": True, "mode": "scratch", "matched_numel": 0})


def best_checkpoint(directory):
    import torch
    directory = Path(directory).resolve()
    last = directory / "last.ckpt"
    if not last.exists():
        raise FileNotFoundError("Training did not write last.ckpt with checkpoint-selection state")
    state = torch.load(last, map_location="cpu", weights_only=False)
    choices = []
    for callback in state.get("callbacks", {}).values():
        if not isinstance(callback, dict) or not callback.get("best_model_path"):
            continue
        candidate = Path(callback["best_model_path"]).resolve()
        score = callback.get("best_model_score")
        if candidate.parent != directory or not candidate.exists() or score is None:
            raise ValueError("Best-validation checkpoint metadata points outside this run or is incomplete")
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("Non-finite checkpoint selection score")
        choices.append({"path": str(candidate), "monitor": "val/loss", "value": score})
    if len(choices) != 1:
        raise ValueError(f"Expected exactly one best-validation checkpoint, got {len(choices)}")
    return choices[0]


def classification(args):
    import torch
    from sklearn.metrics import roc_auc_score, roc_curve
    cfg = read_yaml(args.config)
    batches = torch.load(args.prediction, map_location="cpu", weights_only=False)
    score = torch.cat([batch["classification"][cfg["score_key"]].softmax(-1)[:, cfg["score_index"]]
                       for batch in batches]).numpy()
    process = torch.cat([batch[cfg["target_key"]].reshape(-1) for batch in batches]).numpy()
    target = np.isin(process, cfg["signal_ids"])
    if not np.isin(process, cfg["signal_ids"] + cfg["background_ids"]).all():
        raise ValueError("Unexpected process IDs in classification evaluation")
    weight = torch.cat([batch[cfg["weight_key"]].reshape(-1) for batch in batches]).numpy()
    if not np.isfinite(score).all() or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("ROC probe requires finite scores and nonnegative event weights")
    if weight[target].sum() <= 0 or weight[~target].sum() <= 0:
        raise ValueError("Both signal and background need positive total weight")
    fpr, tpr, _ = roc_curve(target, score, sample_weight=weight, drop_intermediate=False)
    efficiency = float(cfg["signal_efficiency"])
    if not 0 < efficiency < 1:
        raise ValueError("signal_efficiency must be in (0, 1)")
    # First actual threshold attaining the target; no optimistic interpolation.
    index = np.flatnonzero(tpr >= efficiency)[0]
    background_efficiency = float(fpr[index])
    metrics = {"auc": float(roc_auc_score(target, score, sample_weight=weight)),
               "background_efficiency": background_efficiency,
               "achieved_signal_efficiency": float(tpr[index])}
    if background_efficiency > 0:
        metrics["background_rejection"] = 1 / background_efficiency
        metrics["sic"] = float(tpr[index] / np.sqrt(background_efficiency))
    write_json(args.output, {"metrics": metrics, "events": len(target),
                            "signal_efficiency_target": efficiency,
                            "notes": [] if background_efficiency else
                            ["No surviving background: rejection and SIC are undefined, not infinite"]})


def load_quantum_analysis(repository, config):
    """Load the native evaluator with an explicit, portable RooUnfold library."""
    import ROOT
    library = config.get("roounfold_library")
    if library or not all(hasattr(ROOT, key) for key in ("RooUnfoldResponse", "RooUnfoldSvd")):
        resolved = os.path.expandvars(os.path.expanduser(library or "libRooUnfold"))
        if ROOT.gSystem.Load(resolved) < 0 and library:
            raise RuntimeError(f"Cannot load RooUnfold library: {resolved}; set evaluation.roounfold_library")
    repository = Path(repository).resolve()
    sys.path.insert(0, str(repository))
    spec = importlib.util.spec_from_file_location("qe_reference_analysis", repository / "analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key in ("RooUnfoldResponse", "RooUnfoldSvd"):
        if not hasattr(ROOT, key):
            raise RuntimeError(f"Missing RooUnfold class: {key}; set evaluation.roounfold_library")
    return module


def quantum(args):
    import torch
    config = read_yaml(args.config)
    module = load_quantum_analysis(args.repository, config)
    raw = torch.load(args.prediction, map_location="cpu", weights_only=False)
    count = sum(len(batch["full_input_point_cloud"]) for batch in raw)
    truth_d, truth_final, truth_data = module.truth_ana(raw)
    reconstructed, full, cut = module.Get_ana_data(raw, truth_data)
    Path("results/csv").mkdir(parents=True, exist_ok=True)
    final, result, up, down, _ = module.unfolding_process(full, count)
    module.save_unfold_res(final, result, up, down, "probe")
    coefficients = config["trace_coefficients"]
    reconstructed_d = -sum(result[name] for name in coefficients)
    observable = config["observable"]
    uncertainty = 0.5 * (abs(final[observable]["uncertainty_up"]) +
                         abs(final[observable]["uncertainty_down"]))
    write_json(args.output, {"metrics": {
        "trace_abs_bias": float(abs(reconstructed_d - truth_d)),
        "observable_abs_bias": float(abs(final[observable]["value"] - truth_final[observable]["value"])),
        "observable_uncertainty": float(uncertainty),
        "selection_efficiency": float(np.mean(np.asarray(cut))),
    }, "events": count, "observable": observable,
        "truth_trace": float(truth_d), "reconstructed_trace": float(reconstructed_d),
        "notes": ["Native v1 unfolding; uncertainty is its reported statistical uncertainty, not seed variance"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["audit", "best", "classification", "quantum", "script"])
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--weight-source", default="state_dict")
    parser.add_argument("--policy")
    parser.add_argument("--prediction")
    parser.add_argument("--repository")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ray-storage", help="Override native Ray RunConfig storage under this directory")
    args, script_args = parser.parse_known_args()
    if args.operation != "script" and script_args:
        parser.error(f"Unexpected arguments: {script_args}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if args.operation == "audit":
        audit(args)
    elif args.operation == "best":
        write_json(args.output, best_checkpoint(args.checkpoint))
    elif args.operation == "classification":
        classification(args)
    elif args.operation == "quantum":
        quantum(args)
    else:
        if script_args and script_args[0] == "--":
            script_args = script_args[1:]
        if not script_args:
            parser.error("script requires -- script.py [arguments]")
        sys.argv = script_args
        sys.path.insert(0, str(Path(script_args[0]).resolve().parent))
        with ray_storage(args.ray_storage):
            runpy.run_path(script_args[0], run_name="__main__")


if __name__ == "__main__":
    main()
