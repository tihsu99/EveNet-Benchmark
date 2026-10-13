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

For the supplied mixed-runtime workflow, install the two environment-specific
requirements from this checkout on the host (outside Shifter):

First match the host interpreter to the running cluster's exact Python version.
The verified `evenet:1.3` image uses Python 3.12.3 and Ray 2.44.1. For that image:

```bash
conda install -p /pscratch/sd/t/tihsu/env/benchmark 'python=3.12.3'
/pscratch/sd/t/tihsu/env/benchmark/bin/python -c 'import sys, ray; print(sys.version); print(ray.__version__)'
```

After the Conda command succeeds, install/check the Python dependencies below.
This changes the host benchmark environment only; leave the existing `unfold`
interpreter and running Shifter Ray services as they are.

```bash
cd /global/u1/t/tihsu/EveNet-Benchmark
/pscratch/sd/t/tihsu/env/benchmark/bin/python -m pip install -r requirements.txt
/pscratch/sd/t/tihsu/env/benchmark/bin/python -m pip check
/pscratch/sd/t/tihsu/env/unfold/bin/python -m pip install -r requirements-qe.txt
/pscratch/sd/t/tihsu/env/unfold/bin/python -m pip check
/pscratch/sd/t/tihsu/env/benchmark/bin/python -m benchmark.preflight configs/workflow.yaml
```

`requirements.txt` selects `.[host]`: the driver, Higgs evaluation and native AD
prediction/evaluation dependencies, including Ray, Awkward/vector, numdifftools,
JetNet and zfit. Pins follow the
[selected public AD recipe](https://github.com/EveNet-HEP/CMSOpenData-Upsilon-AnomalyDetection/blob/ff4f16845a7919d8c3e49f29240f9d79071e2a64/requirements.txt)
and its documented `zfit==0.22.0` installation. Use Python 3.12 for this profile;
zfit 0.22 does not support Python 3.13. The host Python/Ray versions must match
the existing Shifter Ray cluster, including the Python patch version:
[Ray 2.44.1 checks the complete version by default](https://github.com/ray-project/ray/blob/ray-2.44.1/python/ray/_private/utils.py).
Python 3.12.15 therefore cannot directly join a cluster running Python 3.12.3.
Pip requirements cannot enforce the interpreter patch version. Installing these
pins can change existing package versions; run `pip check` before preflight.

`requirements-qe.txt` selects `.[quantum]`: PyTorch and the Python analysis
dependencies imported by the pinned QE evaluator. Install it into the existing
`unfold` environment; it does not install ROOT or rebuild RooUnfold. On Python
3.14, this profile allows PyTorch 2.9+ (below 3) and a compatible Awkward 2 release:
PyTorch 2.6 and the compiled dependency of Awkward 2.8.1 have no Python 3.14 wheels.
Earlier Python versions retain those recipe pins. QE evaluation runs in its own
process and loads predictions onto the CPU, so it does not join the training Ray
cluster. Keep the Python version used by the existing ROOT build. Training
continues to use the selected EveNet-Full/core Shifter environment. Source paths
for EveNet and the native analysis repositories still come from the workflow.
The host profile covers this benchmark's AD generation-quality probe, not the
separate upstream classifier-training or bump-hunting stages.

If an earlier QE install stopped during dependency resolution, rerun the complete
`requirements-qe.txt` installation after updating this checkout. It also installs
base dependencies such as `huggingface_hub`; `pip check` alone does not install
missing packages or verify that optional extras are complete. Continue to
preflight only after both installation and `pip check` succeed.

The earlier `/usr/lib64/python3.6` error came from the system Python; installing
`tqdm` there would not provide a supported runtime. Use the explicit environment
executables above instead.

W&B runs use `wandb_project` (default `EveNet-Benchmark`) and the team configured
in `adaptation.logger.wandb.entity` (default `ytchou97-university-of-washington`,
matching the public AD/Higgs recipes). Native recipe entities are preserved when
no override is supplied. A task can override the team through
`benchmarks.<task>.overrides.logger.wandb.entity`; set it to `null` to let W&B
resolve its default entity. The entity selects the team workspace, not a W&B run
group. Template run IDs are still removed so adaptations create independent runs.

### Fast environment preflight (no training)

After syncing this checkout and downloading the configured sources/data, run
from the host shell for the supplied workflow (do not prefix this with Shifter):

```bash
/pscratch/sd/t/tihsu/env/benchmark/bin/python -m benchmark.preflight configs/workflow.yaml
```

This creates a fresh `preflight/<timestamp>/report.json` under `download_root`,
plus per-runtime JSON reports and logs. It does not create a benchmark version,
prepare/copy datasets, load checkpoint tensors, or launch training/prediction.
It resolves the same workflow overrides as a real run, checks input globs and
one Parquet header per split, checks segmentation query counts against shape
metadata, and imports the native entry points without calling their main blocks.
Identical runtime checks are reused. All checks are collected before returning:
exit 0 means PASS; exit 1 means at least one failure. Fix every failure first.
The runtime-check timeout defaults to 120 seconds; change it with `--timeout`.

When an existing Ray cluster is ready, optionally add `--ray`. This attaches and
reports capacity without scheduling work or starting/stopping the cluster.
Neither mode tests CUDA kernels, remote worker environments, full dataset
integrity, checkpoint transfer, or physics results. After preflight passes, use
the existing `run --check-only` stage for data verification and CPU checkpoint
transfer auditing before an end-to-end run.

Environment checks are declared in `configs/probe.yaml` under `preflight`, with
task-specific overrides. Training uses `execution.python`; prediction uses
`execution.prediction_python` when set, otherwise the training command; evaluation
uses `execution.analysis_python`. Each task can override these commands under
its own `execution` mapping. Commands remain argv lists; no shell evaluation is
performed. Importing a custom script executes its top-level code, so keep its
training/evaluation entry point behind `if __name__ == "__main__"`.

The supplied workflow keeps the driver/runner, Higgs evaluation, and AD native
prediction/evaluation in `/pscratch/sd/t/tihsu/env/benchmark/bin/python` on the
host. Training and QE/Higgs prediction use Shifter with the `/tmp/ray` mount
described below.
The driver runtime is also used for the inner benchmark runner and Slurm driver;
it must not silently switch back to the training container. Calling a host
environment's Python from inside Shifter still runs it inside that container.
Preflight checks the driver imports, AD `numdifftools`, JetNet/zfit, and the
native scripts in their selected runtimes. It does not install packages.

The default workflow now selects `/pscratch/sd/t/tihsu/env/unfold/bin/python`
for QE evaluation only. Preflight verifies that this executable and its imports
are usable; the path alone does not prove that ROOT/RooUnfold is installed.
The [QE public instructions](https://github.com/EveNet-HEP/TT2L-QC-Study#unfolding-study)
require ROOT and a compiled RooUnfold library in addition to the EveNet image.
The evaluator first tries `libRooUnfold` on the runtime library search path,
then preserves the native loader as a fallback. To specify the Linux library
explicitly, add this to the workflow's existing task override:

```yaml
conditions:
  suite_overrides:
    benchmarks:
      qe:
        execution:
          analysis_python: [/pscratch/sd/t/tihsu/env/unfold/bin/python]
        evaluation:
          roounfold_library: /absolute/path/to/libRooUnfold.so
```

The same loader is checked by preflight and used for actual QE evaluation.
It verifies both `RooUnfoldResponse` and `RooUnfoldSvd` after importing native
analysis. The upstream pinned script also attempts a relative `.dylib` load;
that message alone is not a failure if the configured library supplied both classes.

The [AD installation instructions](https://github.com/EveNet-HEP/CMSOpenData-Upsilon-AnomalyDetection#step-1-installation)
use a Python 3.12 analysis environment, its `requirements.txt`, and a separate
`zfit==0.22.0` installation; EveNet training uses the `avencast1994/evenet:1.5`
image. Configure a different `anomaly.execution.analysis_python` if needed.
Preflight checks imports used by the generation-quality probe, including the
native AD preparation script, JetNet and zfit. It does not certify all optional
dependencies of the full bump-hunting workflow or enforce every pinned package
version. Reports include the executable and available imported-module versions.
Required environment variables are checked for presence without recording values.

## Download, run, monitor and compare

`configs/workflow.yaml` is the single entry point for pinned source repositories,
Hugging Face downloads, benchmark selection, runtime commands and resources.
The default selects QE, the three Higgs head configurations, and anomaly
detection using the pinned `tihsu/AD-Data` SB/SR dataset.

```bash
cd ~/PycharmProjects/EveNet-Benchmark
python3 -m pip install -r requirements.txt
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
          checkpoints/    # Best/last adapted model checkpoints
          ray/            # Ray Train AND Predict persistent results
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

The native-script adapter enforces the probe's `ray/` storage path, including
prediction scripts that otherwise default to `~/ray_results`. Checkpoints remain
under the same probe's `checkpoints/`. The optional pretraining command places
both directories under its explicit `--output` path. Run from any checkout;
these destinations are absolute and do not depend on your shell's directory.
Existing files in home are not moved or deleted. Ray cluster session logs and
object spilling are managed by the externally started cluster (normally in
`/tmp/ray`), independently of these persistent benchmark results.

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

`resources.nodes`, `gpus_per_node`, and `cpus_per_node` describe the capacity
of your existing allocation. `resources.max_parallel` limits the number of
concurrent probes within one checkpoint run. The runner also sums each active
probe's GPU and CPU demand and admits only jobs that fit; a job-count limit alone
would not prevent six 16-GPU jobs from being submitted to a 16-GPU allocation.
The native Ray trainers reserve the actual devices. CPU accounting includes one
trainer CPU per probe and leaves `driver_cpus` outside the probe budget.

Per-probe `overrides.platform` in `configs/probe.yaml` controls worker count and
batch size. Each worker gets one GPU. `resources.cpus_per_worker` supplies the
shared CPU default; a probe can override it. The optional legacy `resources.workers`
is a shared fallback, not the allocation's total GPU count, and per-probe values
take precedence. Planning prints the effective resource and batch table and saves
it as `probe_resources` in `plan.json`; the report also records resolved resources.

| Probe | Training GPUs | Batch per GPU | Global batch |
| --- | ---: | ---: | ---: |
| QE | 16 | 1024 | 16384 |
| Higgs Cls | 4 | 2048 | 8192 |
| Higgs Cls+Asn | 4 | 2048 | 8192 |
| Higgs Cls+Seg | 4 | 2048 | 8192 |
| AD | 4 | 4096 | 16384 |

GPU counts follow [EveNet Table 2](https://arxiv.org/html/2601.17126v1#S2.T2).
The paper does not specify batch sizes; per-GPU batches come from the pinned
[QE recipe](https://github.com/EveNet-HEP/TT2L-QC-Study/blob/ba0f9050cee8ae6eb799dbb59d37a1cc331343d7/examples/Train_example.yaml),
[Higgs recipe](https://github.com/EveNet-HEP/Exotic-Higgs-Study/blob/c248a12749479f4d58778e76e10449a4d1d0292a/configs/train.yaml), and
[AD recipe](https://github.com/EveNet-HEP/CMSOpenData-Upsilon-AnomalyDetection/blob/ff4f16845a7919d8c3e49f29240f9d79071e2a64/config/full_train.yaml).
The public AD recipe uses two GPUs; this suite deliberately uses the paper's four
GPUs while retaining the recipe's per-GPU batch. The global batch assumes one
optimizer update per batch with no gradient accumulation, as in the native trainer.
Higgs epochs and patience follow Sec. IV of the paper. QE and AD retain their
pinned native epoch/patience settings where the paper does not specify values.

| Probe | Max epochs (`epochs`) | Schedule epochs (`total_epochs`) | Early stopping patience | Monitor |
| --- | ---: | ---: | ---: | --- |
| QE | 300 | 300 | 50 | `val/loss` |
| Each Higgs probe | 100 below 99K training events, else 50 | Same as max epochs | 20 | `val/loss` |
| AD | 1000 | 1000 | 1000 | `val/generation-recon` |

The native trainer reads `Training.EarlyStopping`; the separate Higgs
`Training.patience: 20` field is not used by that trainer. The shared probe
explicitly overrides the native nested patience of 10 with the paper's 20.
The AD settings come from its [options.yaml](https://github.com/EveNet-HEP/CMSOpenData-Upsilon-AnomalyDetection/blob/ff4f16845a7919d8c3e49f29240f9d79071e2a64/config/options.yaml).
`epochs` limits trainer execution, while `total_epochs` controls the optimizer
schedule. Explicit `adaptation.options.Training` and per-probe
`overrides.options.Training` values take precedence over native settings,
including nested `EarlyStopping` overrides. A configured `epoch_budget` determines
the final execution and schedule epochs from the prepared training count.
These are public-recipe settings, not a claim of exact paper reproduction:
the fixed subsets, single AD fold and requested generation budget remain in use.

All workflows using `configs/probe.yaml`, including the mini campaign, apply
task-specific `pretrained_overrides` when `--checkpoint` is supplied and
`scratch_overrides` with `--scratch`. These merge after the native recipe and
before shared `adaptation` and task `overrides`, so explicit custom settings
retain precedence. Loading checkpoint tensors alone does not change optimizer
settings. Pretraining itself uses a separate command and is unaffected.

Pretrained learning rates follow
[Sec. II.3 / Table 2](https://arxiv.org/html/2601.17126v1#S2.SS3), with the
Higgs-specific intermediate rate in Sec. IV and AD confirmation in Sec. VI:

| Task | Module | Base LR | Peak LR at configured GPU count |
| --- | --- | ---: | ---: |
| QE (16 GPUs) | GlobalEmbedding, PET, ObjectEncoder | `5e-6` | `2e-5` |
| QE (16 GPUs) | Assignment, TruthGeneration, InvisibleInputProjector | `5e-5` | `2e-4` |
| Higgs (4 GPUs) | PET | `4e-5` | `8e-5` |
| Higgs (4 GPUs) | GlobalEmbedding, ObjectEncoder | `1.2e-4` | `2.4e-4` |
| Higgs (4 GPUs) | Active task heads | `4e-4` | `8e-4` |
| AD (4 GPUs) | PET, ObjectEncoder | `5e-5` | `1e-4` |
| AD (4 GPUs) | GlobalEmbedding, GlobalGeneration, ReconGeneration | `5e-4` | `1e-3` |

All active groups use AdamW. QE's public `Train_example.yaml` is a scratch
recipe; the pretrained encoder reduction is explicit in the probe. The pair
core's InvisibleInputProjector shares an explicit AdamW/rate group with
TruthGeneration, avoiding its default Lion configuration taking precedence.
For AD, the component mapping follows the pinned `config/options.yaml`:
GlobalEmbedding retains its full conditioning rate but has a separate group
from PET, so it cannot override PET's reduced rate. Scratch uses uniform base
rates of `5e-5` (QE), `4e-4` (Higgs), and `5e-4` (AD); no pretrained-only
reductions are applied to random initialization.

The Higgs component mapping for
[Sec. IV](https://arxiv.org/html/2601.17126v1#S4)
follows the pinned Higgs `options_pretrain.yaml`, which
the upstream `Make_script.py` selects for pretrained candidates. The intermediate
modules use a separate optimizer group from PET because the native trainer takes
the first component's learning rate for each group. All active modules remain
trainable; the heavy-resonance task's layer-freezing/adapters are not requested
by this Higgs prescription.

The Higgs task's `epoch_budget` mapping selects 100 epochs below 99,000 prepared
training events and 50 epochs at or above that threshold. It sets both execution
and cosine-schedule epochs after native/explicit Training settings are merged;
a task-level mapping overrides the suite-level mapping. Early stopping uses the
consumed `Training.EarlyStopping.patience: 20`, not the unused flat `patience`.
One-epoch warmup remains a native-recipe choice. Workflows without `epoch_budget`
retain their existing epoch settings. Environment preflight defers this count-based
selection until data preparation. The inherited `train_fraction` applies to the
available training pool after validation splitting; verify the actual prepared
count before identifying it with the paper's 10,000-event point (5% of a
200,000-event reference). These mini models and this probe's metrics
do not constitute a complete reproduction of the paper. Sync the workflow and
the benchmark Python files, then launch with new run names; saved plans and running jobs
are not updated by edits to the workflow.

QE uses `diffusion_every_n_epochs: 100` and
`Components.TruthGeneration.diffusion_steps: 20` under
`benchmarks.qe.overrides.options.Training` in `configs/probe.yaml`.
Training-time validation sampling runs at epochs 100, 200, etc., with 20 sampler
steps; validation batch cadence, diffusion loss and early stopping on `val/loss`
retain their native settings. Diagnostic plots therefore use coarser sampling.
QE's `prediction.overrides.options.Training.Components.TruthGeneration.diffusion_steps`
restores 200 steps for final prediction/evaluation.

`prediction.overrides` merges native configuration fields into the prediction
config only, after resolving the prediction template. Per-probe settings override
shared `prediction.overrides`. The runner still controls prediction batch size,
evaluation data/output paths, dataset limit and checkpoint initialization.
These settings apply to newly launched runs, not a running trainer or a saved plan.

Prediction defaults to four times the training batch per GPU, with the same GPU
worker count for QE and Higgs. These are throughput settings, not paper training
settings, and the larger batches have not been validated on NERSC:

| Probe | Prediction GPUs | Prediction batch per GPU |
| --- | ---: | ---: |
| QE | 16 | 4096 |
| Each Higgs probe | 4 | 8192 |
| AD generation | 1 | 16384 |

Set `conditions.suite_overrides.prediction.batch_size_multiplier` in the workflow
YAML (default `4`). In `probe.yaml`, a probe's `prediction.batch_size_multiplier`
overrides the shared multiplier, and `evaluation.batch_size` takes precedence as
an absolute batch size. Suites without either setting retain the training batch.
Training batches and GPU admission budgets stay unchanged. If a larger inference
batch exceeds device memory, reduce that probe's multiplier or absolute batch.
AD generates 20,000 events by default, controlled independently by
`benchmarks.anomaly.evaluation.generated_events`.

For the default four-node, 16-GPU allocation, QE runs alone; after it finishes,
the three Higgs probes and AD can run together (4 + 4 + 4 + 4 GPUs).
Admission conservatively counts a probe's training resources until its entire
pipeline completes; CPU-only stages may leave GPUs idle. Failures are recorded
per probe and other probes continue. Each probe keeps its own configs, checkpoints,
predictions, and logs. Only the coordinator writes the combined report.

The native AD generator uses `cuda:0` without Ray. The supplied workflow sets
`benchmarks.anomaly.execution.native_on_driver: true`: AD generation and JetNet
evaluation run as host subprocesses while a same-node Ray actor reserves one GPU
and the configured CPUs. The subprocess receives the reserved GPU through
`CUDA_VISIBLE_DEVICES`, and the reservation is released when it exits. Without
this option, these scripts run inside a Ray task as before. Generation keeps the
existing single-rank seed/event semantics; its batch is separately set by the
prediction multiplier or an explicit `evaluation.batch_size`.
The host benchmark environment needs a Python/Ray version compatible with the
existing cluster. `preflight --ray` checks connectivity and driver-node resource
capacity; it does not run GPU kernels or allocate the resource lease.

All workers must see the same code, prepared data and output paths, with compatible
Python/Ray/model dependencies. GPU visibility and container mounts remain part of
the cluster you set up. Ray controls training-worker placement; host-native AD
stages reserve resources on the driver's node. Multiple checkpoint drivers share
Ray's cluster resources, but each driver's local concurrency limit is independent.

For your interactive allocation, first start your existing Ray cluster on all
four nodes using the same Shifter image. `salloc` reserves resources but does not
start Ray. The host driver and containerized Ray processes must see the same
node-local `/tmp/ray`, including the live Unix sockets. Shifter's default `/tmp`
can be separate from the host's. In that case the driver reaches the head's TCP
port but waits for `node_ip_address.json` and cannot access the raylet socket.

Before launching any Ray services in a fresh allocation, create the mount source
on every node, from the host:

```bash
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
  --ntasks-per-node=1 mkdir -p /tmp/ray
```

In every existing head/worker `srun` command, add the same Shifter volume option:

```bash
# Keep the existing srun resources, node selection, and background execution.
shifter --volume=/tmp/ray:/tmp/ray ./start-head.sh "$head_node_ip"
shifter --volume=/tmp/ray:/tmp/ray ./start-worker.sh "$head_node_ip:$port"
```

This assumes the inner scripts use Ray's default `/tmp/ray`; if they set a custom
`--temp-dir`, mount that node-local directory at the same absolute path instead
and update `runtime.python` accordingly. Do not copy the JSON file or put all
nodes' sockets into one shared directory. The supplied workflow includes this
volume option for subsequent training/prediction commands. Mount changes do not
repair already-running Ray processes: use your normal cluster lifecycle to
relaunch when existing workloads are finished. The benchmark never stops Ray.
See [NERSC's volume-mount documentation](https://docs.nersc.gov/development/containers/shifter/how-to-use/#volume-mounting-in-shifter).

On the driver node, verify that the host sees the newly started session and
both sockets before testing the connection:

```bash
ls -l /tmp/ray/session_latest/node_ip_address.json /tmp/ray/session_latest/sockets
/pscratch/sd/t/tihsu/env/benchmark/bin/python -m benchmark.preflight configs/workflow.yaml --ray
```

Run this from a Ray GPU node, with the existing cluster address set as below.
Proceed only after the Ray preflight passes. Then run from the benchmark checkout:

```bash
export RAY_ADDRESS=HEAD_IP:6379
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
shifter --volume=/tmp/ray:/tmp/ray ray status --address="$RAY_ADDRESS"
# Confirm 4 alive nodes, 16 GPUs, and enough CPUs (QE needs 481).
/pscratch/sd/t/tihsu/env/benchmark/bin/python -m benchmark.workflow run configs/workflow.yaml --name plan-preview --dry-run
/pscratch/sd/t/tihsu/env/benchmark/bin/python -m benchmark.workflow run configs/workflow.yaml --name candidate-parallel
```

Replace `HEAD_IP` with the cluster head address. The default CPU budget assumes
Ray advertises 128 logical CPUs per node. If your cluster advertises fewer CPUs,
adjust the CPU capacity and per-worker CPU requests together; GPU counts and
batch sizes need not change. Keep resource settings fixed across candidates.

If using your external Ray job submitter, submit one benchmark workflow per
checkpoint and let its internal scheduler launch probes. The
[Ray Jobs endpoint](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html)
is `http://HEAD_IP:8265`, while `RAY_ADDRESS=HEAD_IP:6379` connects Python drivers
to the cluster. Do not reserve training GPUs on the job entrypoint: native Ray
Train workers request them separately. Your shell throttle controls active
checkpoint workflows, not their GPU allocation. Check each background command's
exit status; a final bare `wait` does not reliably report earlier job failures.

With the supplied mixed-runtime workflow, launch the driver on the host of an
existing Ray GPU node and let `runtime.python` enter the training container with
the same Ray volume mount. Keep `runtime.image: null` so the driver stays on the host.
A fully containerized workflow may instead use a Shifter driver and plain
`python3` child commands, but it cannot run host-only analysis environments.
Alternatively, `--backend slurm` submits exactly one CPU
driver task using the configured account, partition, time and `driver_cpus`.
It exports your Ray address. `runtime.image` adds the image to the SBATCH header
and executes the driver through Shifter. This optional submission does not create
a Ray cluster. Host AD stages require the submitted driver to land on an existing
Ray GPU node; an unrelated CPU node is rejected. Cancelling that driver does not stop your externally managed Ray
services; use your normal Ray job/cluster controls for any surviving training work.

```bash
evenet-benchmark run configs/workflow.yaml --name script-preview --backend slurm --dry-run
evenet-benchmark run configs/workflow.yaml --name slurm-v1 --backend slurm
```

Anomaly is enabled through `conditions.benchmarks` and `downloads.ANOMALY_DATA`.
For an existing local dataset, `variables.ANOMALY_DATA` can override that path.
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

Generated model configs keep `event_info` in a sibling file such as
`train.event_info.yaml`, referenced through EveNet's native `default:` syntax.
Keep that file beside `train.yaml` (likewise for prediction). This allows Ray
workers to reload the config after the driver has constructed `EventInfo`;
flattening it inline can cause `AttributeError: 'EventInfo' object has no attribute 'merge'`.
After updating the runner, rerun with a new checkpoint name to regenerate configs;
existing failed run artifacts are retained.

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

The Higgs Cls+Seg recipe sets `Segmentation.num_queries: 3` through that override,
matching its public target slots. The candidate's backbone and other segmentation
settings are preserved. A global pretraining head with four queries cannot be
used unchanged with this dataset's three-slot segmentation truth.

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
classification metrics, report comparability, CPU/GPU-bounded parallel dispatch,
failure isolation, and native AD GPU reservations. They are not evidence of
successful NERSC training, QE ROOT execution, anomaly generation, or physics
improvement. Run the three native adapters against your data before treating the
reports as production validated. A real Hub smoke download retrieved only the
pinned public model README; full dataset downloads and GPU/multi-node execution
have not been run here. Bootstrap uncertainty and multi-seed robustness
are not estimated by this first version; native QE statistical uncertainty is
reported separately. Calibration magnitude alone cannot establish anomaly
sensitivity.
