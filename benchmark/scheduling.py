"""Bound concurrent probes by their resolved training CPU/GPU requirements."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .common import merge, path_value, resolve_template


def configured_platform(suite, task, base):
    platform = merge(suite.get("adaptation", {}).get("platform", {}),
                     task.get("overrides", {}).get("platform", {}))
    if not {"number_of_workers", "resources_per_worker", "batch_size"} <= platform.keys():
        repository = path_value(task["repository"], base)
        native = resolve_template(path_value(task["train_yaml"], base), repository,
                                  task.get("template_tokens"))
        platform = merge(native["platform"], platform)
    return platform


def task_resources(platform):
    workers = int(platform["number_of_workers"])
    per_worker = platform["resources_per_worker"]
    cpus = int(per_worker["CPU"])
    gpus = per_worker.get("GPU", 1 if platform.get("use_gpu", True) else 0)
    batch = int(platform["batch_size"])
    if min(workers, cpus, batch) < 1 or gpus not in (0, 1):
        raise ValueError("Probes require positive worker/CPU/batch counts and 0 or 1 GPU per worker")
    if bool(gpus) != bool(platform.get("use_gpu", True)):
        raise ValueError("use_gpu must agree with resources_per_worker.GPU")
    return {"CPU": workers * cpus + 1, "GPU": workers * gpus,
            "workers": workers, "batch_per_worker": batch, "global_batch": workers * batch}


def run_parallel(items, run, demands, config, on_start, cancel_event=None):
    """Yield completed futures; only the caller writes the aggregate report."""
    max_jobs = int(config.get("max_jobs", 1))
    if max_jobs < 1:
        raise ValueError("execution.parallel.max_jobs must be positive")
    budget = {key: int(config.get(key, max(d[key] for d in demands.values())))
              for key in ("CPU", "GPU")}
    for name, demand in demands.items():
        if any(demand[key] > budget[key] for key in budget):
            raise ValueError(f"Probe {name} needs {demand}; allocation budget is {budget}")
    pending = list(items)
    running = {}
    used = {key: 0 for key in budget}
    pool = ThreadPoolExecutor(max_workers=max_jobs)
    try:
        while pending or running:
            for item in list(pending):
                name, _ = item
                demand = demands[name]
                if len(running) >= max_jobs:
                    break
                if any(used[key] + demand[key] > budget[key] for key in budget):
                    continue
                on_start(name)
                future = pool.submit(run, *item)
                running[future] = name
                for key in used:
                    used[key] += demand[key]
                pending.remove(item)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                name = running.pop(future)
                for key in used:
                    used[key] -= demands[name][key]
                yield name, future
    finally:
        if running and cancel_event is not None:
            cancel_event.set()
        pool.shutdown(wait=True, cancel_futures=True)
