# EveNet-Benchmark

Fixed-data, fixed-budget probes of EveNet foundation-model checkpoints. The
pipeline runs independent downstream adaptations, writes one `report.json` per
candidate, and renders a scouting report or a multi-checkpoint comparison.

This repository orchestrates the existing physics repositories. It does not copy
the model core or reimplement QE unfolding/anomaly calibration. Grid Study is
disabled: sharing the `evenet` import is possible, but the public EveNet-Lite
backbone does not construct or forward the pair states used by the new model.

## Scope

| Probe | Adaptation data | Evaluation | Main report |
| --- | --- | --- | --- |
| QE | Fixed 5% of the training split | Full fixed evaluation split, native v1 unfolding | Trace bias, concurrence bias/uncertainty, selection efficiency |
| Exotic Higgs: Cls, Cls+Asn, Cls+Seg | Shared fixed 5% after separating validation | Shared full fixed evaluation split | Weighted AUC, rejection and SIC for each head configuration |
| Upsilon anomaly | One chosen fold from K >= 2 | Fixed SR data and generated-event budget | Calibration magnitude, coverage, native JetNet MMD, selection efficiency |
| Grid | Disabled | Not run | Explicit skipped status |

The full report has 17 physics metrics in three benchmark groups.
Exotic Higgs starts with the explicit `ma=30` hypothesis in YAML; add another
named benchmark entry for another hypothesis. It is a classification probe and
does not reproduce the full v1 limit fit. Anomaly is a generation-quality probe,
not a measurement of bump-hunt significance. Smaller uncertainty is useful only
alongside acceptable bias and selection efficiency.

The three Higgs configurations share one `data_group`, materialized once, and
start independently from the same checkpoint with the same adaptation budget.
Their explicit progressive schedule gives every enabled classification,
assignment or segmentation objective a nonzero loss weight. A shared data group
with differing input/split settings is rejected.

## Runtime and source repositories

Python 3.10+ is sufficient for orchestration; use the Python version required by
the selected EveNet runtime (the public examples use Python 3.12).

```bash
cd ~/PycharmProjects/EveNet-Benchmark
python3 -m pip install -e '.[classification,test]'
```

Training additionally requires the selected EveNet-Full/core environment
(PyTorch, Lightning, Ray, and its dependencies). QE evaluation needs the working
ROOT/RooUnfold environment and the library path expected by the native
`analysis_core/unfold.py`. Anomaly evaluation needs that repository's fitting,
Awkward/vector, and JetNet dependencies. This package does not install or modify
these external stacks.

## Download, run, monitor and compare

`configs/workflow.yaml` is the single entry point for pinned source repositories,
Hugging Face downloads, benchmark selection, runtime commands and resources.
The default selects QE and the three Higgs head configurations. Public prepared
anomaly SB/SR data was not found in the inspected Hub repositories, so anomaly
is excluded from this download example until its prepared data is supplied.

```bash
cd ~/PycharmProjects/EveNet-Benchmark
python3 -m pip install -e '.[classification]'
# Review selected files/bytes, then download once.
evenet-benchmark download configs/workflow.yaml --dry-run
evenet-benchmark download configs/workflow.yaml

# Start your Slurm/Shifter Ray cluster separately, then attach its address.
export RAY_ADDRESS=HEAD_IP:6379
evenet-benchmark run configs/workflow.yaml --name v1-audit --check-only
evenet-benchmark run configs/workflow.yaml --name v1
evenet-benchmark run configs/workflow.yaml --name candidate \
  --checkpoint /shared/candidate.ckpt --network /shared/candidate-network.yaml
evenet-benchmark status configs/workflow.yaml --name candidate --watch
evenet-benchmark compare configs/workflow.yaml
```

Set `download_root` in the workflow YAML to choose the storage root for data
and all evaluation results:

```yaml
download_root: /shared/your-storage/evenet
```

The supplied configuration produces this layout (`<name>` is the run's `--name`):

```text
<download_root>/
  data/
    qe/                   # Downloaded QE dataset
    higgs/                # Downloaded Higgs dataset
    models/               # Downloaded reference checkpoints
    prepared/             # Shared, verified probe subsets
  version/
    <name>/
      job/                # Plan, resolved suite, status, driver and Slurm logs
      results/
        candidate.ckpt    # Snapshot of the evaluated checkpoint (unless scratch)
        report.json
        figures/
        <probe>/          # Adapted checkpoints, predictions, metrics and logs
  sources/                # Downloaded source repositories
  assets.lock.json
  comparison/             # Combined checkpoint comparison
```

Relative `download_root` values resolve against the workflow YAML directory;
environment variables such as `${SCRATCH}/evenet` are supported. `workdir` is
accepted as a fallback only when `download_root` is absent. Each download's
`local_dir` is relative to `data/`; an absolute value can explicitly override it.
Resolved locations automatically feed the probe configuration. Use
`download --dry-run` to inspect destinations before downloading.

Run `status --watch` from another terminal during execution. The driver updates
`version/<name>/job/status.json`, and each probe updates
`version/<name>/results/<probe>/progress.json`. Status and comparison commands
find runs under `version/` automatically. Downloads and probes show progress,
native output, and periodic heartbeat messages in terminal output and logs.
Use a distinct name per invocation. `run --dry-run` writes a reviewable plan
(and Slurm script when selected), reserving that name.

Adding a new download/source entry reuses the existing storage and pinned
revisions; rerun `download` to fetch the new entry. Changing or removing an
existing locked entry, or changing prepared data selection, requires a new
`download_root`. Each new run plan saves its own asset lock snapshot, so later
additions do not invalidate that plan. Existing files from the previous `assets/`, `jobs/`, and
`runs/` layout are not moved automatically. Older plans can still be inspected
with `status /absolute/path/to/plan.json`; use `compare --reports` to explicitly
select older report files.

The default weights and network come from the public v1 release. To probe a new
pair-representation checkpoint, set `sources.EVENET_CORE.path` to its matching
core checkout and supply its actual network YAML. The transfer audit rejects
incompatible or silently dropped pair parameters. A matching runtime with the
native training/physics dependencies is still required; installing this wrapper
alone does not provide a complete QE ROOT/RooUnfold environment.

### Existing Ray cluster and optional Slurm driver

The benchmark never issues `ray start` or `ray stop`. `ray.address: auto` attaches
to your existing cluster, preferring `RAY_ADDRESS`; an explicit YAML address is
also accepted. `ray.address: null` with no environment address allows the native
single-node runtime to initialize Ray. Multi-node execution requires an existing
cluster. The driver checks alive nodes and aggregate CPU/GPU capacity, waits for
available resources, and disconnects its own client without stopping the cluster.

`resources.nodes`, `gpus_per_node`, `workers` and `cpus_per_worker` describe the
training demand on that existing Ray cluster. They feed the native EveNet Ray
worker configuration. They do not allocate nodes or enforce worker placement on
each node. Set the same worker count for all candidates in a controlled study.
All workers must see the same code, prepared data and output paths, with compatible
Python/Ray/model dependencies. GPU visibility and container mounts remain part of
the cluster you set up.

Inside an existing Shifter allocation, use `runtime.driver: [shifter, python3]`
with `backend: local` to launch the driver once. The commands inside that driver
can remain `runtime.python: [python3]`; `analysis_python` can select a separate
physics environment. Alternatively, `--backend slurm` submits exactly one CPU
driver task using the configured account, partition, time and `driver_cpus`.
It exports your Ray address. `runtime.image` adds the image to the SBATCH header
and executes the driver through Shifter. This optional submission does not create
a Ray cluster. Cancelling that driver does not stop your externally managed Ray
services; use your normal Ray job/cluster controls for any surviving training work.

```bash
evenet-benchmark run configs/workflow.yaml --name script-preview --backend slurm --dry-run
evenet-benchmark run configs/workflow.yaml --name slurm-v1 --backend slurm
```

To enable anomaly, add `anomaly` to `conditions.benchmarks` and supply
`variables.ANOMALY_DATA`, or a matching `downloads.ANOMALY_DATA` Hub entry.
The native prepared files required below must already exist in that layout.
Grid remains explicitly skipped.

Public sources inspected for the initial adapters:

- [TT2L-QC-Study](https://github.com/EveNet-HEP/TT2L-QC-Study):
  `examples/Train_example.yaml`, `examples/Predict_example.yaml`, and `analysis.py`.
- [Exotic-Higgs-Study](https://github.com/EveNet-HEP/Exotic-Higgs-Study):
  `configs/train.yaml`, `configs/predict.yaml`, and the score/weight convention in
  `Produce_ntuple.py`.
- [CMSOpenData-Upsilon-AnomalyDetection](https://github.com/EveNet-HEP/CMSOpenData-Upsilon-AnomalyDetection):
  `01predict_SR_certain_fold.py`, `02prepare_classification_dataset.py`.
- [EveNet-Lite backbone](https://github.com/EveNet-HEP/EveNet-Lite/blob/main/evenet_lite/model.py):
  the missing pair-state wiring is why Grid is skipped rather than silently
  dropping pair weights.

## Configure once

Copy `configs/probe.yaml` to `configs/local.yaml`. Paths may be absolute,
relative to the suite YAML, or environment variables. No data or checkpoint
paths are inferred from old runs.

```bash
cp configs/probe.yaml configs/local.yaml
export EVENET_CORE=/path/to/current/EveNet-Full
export QE_REPO=/path/to/TT2L-QC-Study
export QE_DATA=/path/to/TT2L-dataset
export HIGGS_REPO=/path/to/Exotic-Higgs-Study
export HIGGS_TRAIN_DATA=/path/to/evenet-train/evenet-ma30
export HIGGS_TEST_DATA=/path/to/evenet-test/evenet-ma30
export ANOMALY_REPO=/path/to/CMSOpenData-Upsilon-AnomalyDetection
export ANOMALY_DATA=/path/to/prepared-anomaly-data
```

`ANOMALY_DATA` must already contain the native prepared `SB/data.parquet`,
`SR/data.parquet`, `SB/normalization.pt`, and `SB/shape_metadata.json`.
The anomaly workflow's event-info YAML must describe those exact features.
Change the explicit signal/background IDs in the Higgs evaluator if your
prediction convention differs from v1.

Set `execution.python` and `execution.analysis_python` to argv lists for your
environments. For example, training can use
`[shifter, "--image=docker:avencast1994/evenet:1.5", python3]` while evaluation
uses a separate Python providing ROOT. Install this package in both environments,
or ensure its directory is mounted at the same path: the runner exports it and
the selected core through `PYTHONPATH`. The core training entry point still
requires the user's `WANDB_API_KEY` environment variable, even with offline
logging. Keys are never written by this runner.

## Prepare immutable subsets

```bash
python3 -m benchmark.prepare configs/local.yaml --output prepared/probe-v1
```

Selection happens once, before training. The preparer hashes each source file,
uses a seed derived from that hash plus the suite seed, separates validation
first, and retains `floor(fraction * eligible_rows)` per training file. The
remaining validation/evaluation rows are retained in full. Thus 5% is per-file
sampling of the eligible training pool, with integer rounding; it is not a
prefix of shuffled files and does not promise class-stratified sampling.

Each task directory contains `train/`, `validation/`, `evaluation/`, selected
row-index arrays, and `manifest.json`. Prediction/training consumers are passed
the materialized directories with `dataset_limit: 1.0`, so they cannot randomly
choose a different 5%. Payloads and manifests are checked before a run.
Existing output directories are never overwritten. A failed preparation has no
valid manifest; use a fresh destination after correcting its inputs.

Set `event_id_columns` to genuine globally unique ID columns (include process/run
identifiers where necessary) to detect overlap across repacked files. Without
these columns, identity is **source content hash + row index**; identical source
files are rejected, but duplicate physical events in differently packed files
cannot be detected. Event-ID validation currently keeps ID hashes in memory.
The official normalization assets are copied unchanged and hashed; the 5% limit
applies to gradient-training examples, not to the original estimation of those
shared normalization statistics. Supply independently fitted training-only
normalization assets if that stricter protocol is required.

Anomaly `fold: {count: 5, index: 0}` selects one fixed deterministic per-file fold.
It is not the original Ray row-order split and is not presented as the historical
paper's exact fold. Never set K=1. An explicit validation file list can be used
for QE/Higgs; the built-in anomaly generation adapter currently expects a named
fold. Keep the ultimate test set sealed when repeatedly selecting checkpoints:
point `evaluation` at a separate fixed probe holdout for model development.

## Optional small pretraining

The `pretraining` section references your complete pretraining YAML and defines
a separate fixed data budget. The example uses 10% of training data and 5 epochs;
these are configurable screening settings, not calibrated scientific defaults.

```bash
export PRETRAIN_YAML=/path/to/pretraining.yaml
export PRETRAIN_DATA=/path/to/pretraining/train
export PRETRAIN_VAL_DATA=/path/to/pretraining/validation
python3 -m benchmark.prepare configs/local.yaml --pretraining --output prepared/pretrain-v1
python3 -m benchmark.pretrain configs/local.yaml \
  --prepared prepared/pretrain-v1 --output runs/pretrain-variant-a
```

This preserves the selected pretraining objectives, initializes a fresh model,
and records the resolved configuration, source revisions, dataset ID, and best
validation checkpoint in `pretraining.json` and `best.json`. It does not choose
the most recently modified file. Keep data, architecture, batch size, and epoch
budget fixed when ablating loss terms. Different pretraining recipes may be
compared under the same downstream protocol.

## Probe an existing checkpoint

```bash
python3 -m benchmark.run configs/local.yaml \
  --prepared prepared/probe-v1 \
  --checkpoint /path/to/candidate.ckpt \
  --network /path/to/candidate-pretraining.yaml \
  --name pair-contrastive --output runs/pair-contrastive
```

`--network` accepts a standalone network YAML or the full pretraining YAML with
its `network.default` and overrides. Architecture is not guessed from tensors.
Task-specific `network_overrides` can adapt feature indices/dimensions; they are
part of the benchmark protocol. For a matched scratch baseline, use the same
network file and replace `--checkpoint ...` with `--scratch`. Use the v1 network
file with the v1 checkpoint when comparing architecture variants.

Before GPU adaptation, a CPU model is constructed using the actual target task
configuration. The audit checks required backbone keys in **both directions**:
missing target parameters, shape mismatches, and unused source pair parameters
are failures. Task heads can initialize independently. Narrow exceptions may be
configured in `load_policy`; inspect the exact `load_audit.json` first. The audit
writes a filtered `initialization.ckpt` so the actual training load matches the
audited tensors. `--weight-source ema_state_dict` explicitly selects EMA source
weights; absence of that state is an error. The adapted model is evaluated with
raw best-validation weights consistently across the three adapters.

Add `--check-only` with a separate output directory to perform data validation
and the CPU transfer audit without starting adaptation. Its report status is
`checked`, never `complete`, and it contains no physics scores.

Each task starts independently from the candidate, not from another task's
adapted checkpoint. The candidate is copied once and hash-checked before any
task, so an ongoing pretraining job cannot replace its weights between probes.
The runner disables pretraining-only pair contrastive loss
and pair monitoring during adaptation. The example uses a fixed epoch budget
with early stopping beyond that budget, fixed batch size/workers, and the task's
native optimizer settings. The core controls its stochastic training behavior;
this is not a claim of bitwise deterministic GPU execution. If a YAML change
enables a new head/objective, verify its optimizer and progressive-loss settings.

Output structure:

```text
runs/pair-contrastive/
  report.json
  qe/
    train.yaml, predict.yaml, evaluation.yaml
    load_audit.json, initialization.ckpt, best.json
    commands.json, audit.log, adapt.log, predict.log, evaluate.log
    checkpoints/, predictions/, metrics.json
  exotic_higgs_cls/...
  exotic_higgs_asn/...
  exotic_higgs_seg/...
  anomaly/...
```

Failures have an explicit status and error; completed tasks are retained. A
partial run returns a nonzero exit status. The report uses strict JSON (no NaN
or Infinity). If no background survives a Higgs threshold, rejection/SIC are
unavailable with an explanatory note, not infinite. Anomaly calibration values
are taken once from `before cut`; the public code reuses them after cuts. The
native anomaly generator fixes rank-zero sampling seed to 42; other script RNGs
are seeded by the wrapper. Event counts are checked against the fixed manifest
for QE and classification prediction.

## Plot one candidate or compare several

```bash
python3 -m benchmark.plot runs/pair-contrastive/report.json \
  --reference configs/paper_reference.yaml --output reports/pair-contrastive
python3 -m benchmark.plot runs/scratch/report.json runs/v1/report.json \
  runs/pair-contrastive/report.json --reference configs/paper_reference.yaml \
  --focus pair-contrastive --output reports/comparison
```

The primary outputs are scientific profiles: `comparison.png`, `comparison.pdf`
and editable vector `comparison.svg`. `report.html` is the data/provenance appendix;
`comparison.csv` and the exact `reference.yaml` snapshot retain the plotted values.

The figure starts with an NMI-style summary table, followed by small-multiple
dot plots in original units. Typography and table styling follow Table 1 on page 4
of the supplied `42256_2026_1322_Author.pdf`: sans-serif table text, a serif title,
warm grey header/alternating rows, thin horizontal rules and no vertical table rules.
`plot_style` in the reference YAML controls font preference lists, scale, table
colors, rule width, model colors/markers and head line styles. Installed Graphik
Naturel/Harding fonts are preferred; this environment renders Arial/Times New Roman
fallbacks. No proprietary fonts are bundled or extracted from the paper.

Each metric has its own focused numeric scale. Open
diamonds and dashed lines mark the reference; candidate markers have distinct shapes
and colors. Light bands show explicitly sourced reference intervals. All candidates
have directly printed values. The Higgs variants share an axis per metric with
separate head-specific reference centers and bands. The same design works for
one to four checkpoints. Table 1 has one column per checkpoint plus a separate
reference column; its width grows to keep four columns readable. `--focus` remains
accepted for compatibility but never hides other checkpoint columns. Metric order,
labels, units, preference direction and `headline: true` come
from the report's YAML display definitions. A reference task may also specify its
headline metric key, including for older reports.

`Paper Results` is the default reference, configured in `configs/paper_reference.yaml`.
It uses the archived fully pretrained EveNet results: QE at 2950 K training events,
Higgs at mass 30 GeV and 100% training, and the anomaly OS ensemble median. The YAML
records selections, source paths/hashes, aggregation and limitations. These are
external scientific context, not a controlled ablation: paper budgets, selections,
head configurations and ensemble aggregation can differ from the quick probes.
The Higgs SIC reference is derived at 70% signal efficiency, not maximum SIC.
Its SIC/rejection bands propagate the stored native FPR uncertainty at fixed TPR,
using the original first-order formulas; they omit signal-efficiency and model
uncertainty. No AUC interval is available.
QE bias and selection efficiency are unavailable in the archived table and stay N/A.
The anomaly bands are the 16th to 84th percentiles of eight archived ensemble entries;
they show ensemble spread, not confidence intervals on the median or candidate seed
uncertainty. Other intervals remain unavailable unless a source provides them.
The QE uncertainty value is itself a metric, not an error bar on that metric.
To add an interval, place `intervals: {metric_key: {low: ..., high: ..., label: ...}}`
under its reference benchmark, and state its statistical meaning and source in the
context. Intervals must contain the reference center and have finite bounds.
`benchmark_order` in the reference YAML controls panel order for archived reports too.
Change the reference YAML when changing the model variant, mass, channel or metric
definition. No external repository is required to render the stored reference values.

For a matched-budget checkpoint baseline, use `--baseline v1` instead of `--reference`.
Candidate protocol IDs must still match, including prepared data, adaptation settings,
evaluator recipes and code provenance. An external reference does not relax this rule.
Table cells show `value (relative improvement %)`, with only the percentage green
for improvement or red for regression. Higher-is-better uses
`100 * (value - reference) / abs(reference)`; lower-is-better reverses the sign.
Zero or missing references and unspecified directions give an unavailable percentage.
These are relative percentages, not percentage-point changes. Lower plots show raw
values only; the separate delta column is removed. Number precision and scientific
notation are controlled by `plot_style` in YAML. Percentages are descriptive, not
significance claims. Raw deltas remain in CSV for compatibility. There is no
composite score, arbitrary radar normalization or
invented confidence interval. Failed or nonfinite candidate metrics show N/A, never
zero. Synthetic demonstrations remain labelled on the figures and in the appendix.

## Validation and limits

```bash
python3 -m pytest tests -q
```

Tests use synthetic Parquet, tiny tensors/checkpoints, and mock stage execution.
They exercise fixed selection, split isolation, changed-input detection, shared
Higgs data and enabled loss objectives, external-Ray capacity checks, one-driver
Slurm script generation, progress/failure reporting, Hub revision/file selection,
checkpoint transfer, validation-based selection, argument forwarding, weighted
classification metrics, and report comparability. They are not evidence of
successful NERSC training, QE ROOT execution, anomaly generation, or physics
improvement. Run the three native adapters against your data before treating the
reports as production validated. A real Hub smoke download retrieved only the
pinned public model README; full dataset downloads and GPU/multi-node execution
have not been run here. Bootstrap uncertainty and multi-seed robustness
are not estimated by this first version; native QE statistical uncertainty is
reported separately. Calibration magnitude alone cannot establish anomaly
sensitivity.
