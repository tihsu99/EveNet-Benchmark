"""Materialize fixed train/validation/evaluation rows before any adaptation."""

import argparse
import glob
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .common import digest, file_hash, load_suite, path_value, write_json


def split_indices(size, seed, fraction, validation_fraction=None, fold=None):
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    order = np.random.default_rng(seed).permutation(size)
    if fold is not None:
        count, index = int(fold["count"]), int(fold["index"])
        if count < 2 or not 0 <= index < count:
            raise ValueError("Keep K >= 2 and select one fold index; K=1 has no training data")
        start, end = size * index // count, size * (index + 1) // count
        validation = order[start:end]
        training = np.concatenate((order[:start], order[end:]))
    elif validation_fraction is not None:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        count = int(size * validation_fraction)
        validation, training = order[:count], order[count:]
    else:
        validation, training = np.array([], dtype=np.int64), order
    return np.sort(training[:int(len(training) * fraction)]), np.sort(validation)


def write_rows(source, indices, destination):
    """Stream full payloads; only row indices are kept in memory per source file."""
    source = pq.ParquetFile(source)
    offset = 0
    with pq.ParquetWriter(destination, source.schema_arrow) as writer:
        for batch in source.iter_batches(batch_size=65536):
            lo, hi = np.searchsorted(indices, [offset, offset + len(batch)])
            if hi > lo:
                writer.write_table(pa.Table.from_batches([batch]).take(indices[lo:hi] - offset))
            offset += len(batch)


def prepare_task(name, task, base, output, seed, require_evaluation=True):
    data = task["data"]
    fraction = float(data.get("train_fraction", 1))
    if data.get("validation") and (data.get("validation_fraction") or data.get("fold")):
        raise ValueError("Choose explicit validation files or an internal split, not both")
    if not data.get("validation") and not data.get("validation_fraction") and not data.get("fold"):
        raise ValueError("An explicit validation set or split is required")
    if data.get("fold") and data.get("validation_fraction"):
        raise ValueError("Choose either fold or validation_fraction")
    destination = Path(output) / name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a prepared dataset: {destination}")
    destination.mkdir(parents=True)
    manifest = {"schema_version": 1, "benchmark": name, "seed": seed,
                "selection": data, "identity": "source SHA256 plus row index",
                "splits": {split: [] for split in ("train", "validation", "evaluation")}}
    seen_files = set()
    seen_ids = set()
    identity_columns = data.get("event_id_columns", [])
    if identity_columns:
        manifest["identity"] = {"columns": identity_columns}
    for split in ("train", "validation", "evaluation"):
        patterns = data.get(split, [])
        if isinstance(patterns, str):
            patterns = [patterns]
        files = sorted({Path(item).resolve() for pattern in patterns
                        for item in glob.glob(str(path_value(pattern, base)), recursive=True)})
        if patterns and not files:
            raise FileNotFoundError(f"No {split} files for {name}: {patterns}")
        if (split == "train" or (split == "evaluation" and require_evaluation)) and not files:
            raise ValueError(f"Missing {split} input for {name}")
        for source in tqdm(files, desc=f"{name}: {split}", unit="file", disable=not sys.stderr.isatty()):
            fingerprint = file_hash(source)
            if fingerprint in seen_files:
                raise ValueError(f"Duplicate or overlapping source file: {source}")
            seen_files.add(fingerprint)
            size = pq.ParquetFile(source).metadata.num_rows
            if identity_columns:
                # Explicit IDs detect overlap even when files have been repacked.
                for batch in pq.ParquetFile(source).iter_batches(columns=identity_columns):
                    for row in batch.to_pylist():
                        if any(value is None for value in row.values()):
                            raise ValueError("Event IDs must not contain nulls")
                        key = digest(row)
                        if key in seen_ids:
                            raise ValueError(f"Duplicate event identity in {source}: {row}")
                        seen_ids.add(key)
            if split == "train":
                file_seed = int(digest([seed, fingerprint])[:16], 16)
                train, validation = split_indices(size, file_seed, fraction,
                                                 data.get("validation_fraction"), data.get("fold"))
                selections = {"train": train}
                if data.get("validation_fraction") or data.get("fold"):
                    selections["validation"] = validation
            else:
                selections = {split: np.arange(size, dtype=np.int64)}
            for target, indices in selections.items():
                if not len(indices):
                    continue
                folder = destination / target
                folder.mkdir(exist_ok=True)
                stem = f"{len(manifest['splits'][target]):05d}-{fingerprint[:12]}"
                index_path, payload = folder / f"{stem}.npy", folder / f"{stem}.parquet"
                np.save(index_path, indices, allow_pickle=False)
                write_rows(source, indices, payload)
                manifest["splits"][target].append({
                    "source": str(source), "source_sha256": fingerprint, "source_rows": size,
                    "indices": str(index_path.relative_to(destination)),
                    "indices_sha256": file_hash(index_path), "rows": len(indices),
                    "payload": str(payload.relative_to(destination)), "payload_sha256": file_hash(payload),
                })
    manifest["counts"] = {split: sum(item["rows"] for item in entries)
                          for split, entries in manifest["splits"].items()}
    required = ("train", "validation", "evaluation") if require_evaluation else ("train", "validation")
    if not all(manifest["counts"][split] for split in required):
        raise ValueError(f"Empty prepared split: {manifest['counts']}")
    assets = {}
    for asset_name, value in data.get("assets", {}).items():
        if Path(asset_name).name != asset_name:
            raise ValueError("Asset names must be filenames")
        source = path_value(value, base)
        assets[asset_name] = {"source": str(source), "sha256": file_hash(source)}
        for split in manifest["splits"]:
            if manifest["counts"][split]:
                shutil.copyfile(source, destination / split / asset_name)
    manifest["assets"] = assets
    manifest["dataset_id"] = digest(manifest)
    write_json(destination / "manifest.json", manifest)
    return manifest


def verify_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = manifest.pop("dataset_id")
    if digest(manifest) != expected:
        raise ValueError("Dataset manifest was modified")
    manifest["dataset_id"] = expected
    for split, entries in manifest["splits"].items():
        for item in entries:
            for key in ("indices", "payload"):
                if file_hash(directory / item[key]) != item[f"{key}_sha256"]:
                    raise ValueError(f"Prepared {key} changed: {item[key]}")
        for name, asset in (manifest["assets"].items() if entries else []):
            if file_hash(directory / split / name) != asset["sha256"]:
                raise ValueError(f"Prepared asset changed: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretraining", action="store_true")
    args = parser.parse_args()
    suite, base = load_suite(args.suite)
    tasks = {"pretraining": suite["pretraining"]} if args.pretraining else suite["benchmarks"]
    prepared_groups = set()
    for name, task in tasks.items():
        group = task.get("data_group", name)
        if task.get("enabled", True) and group not in prepared_groups:
            result = prepare_task(group, task, base, Path(args.output).resolve(), suite["seed"],
                                  require_evaluation=not args.pretraining)
            prepared_groups.add(group)
            print(group, result["counts"], result["dataset_id"])


if __name__ == "__main__":
    main()
