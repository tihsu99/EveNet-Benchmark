"""Submit one driver to the user's existing Ray cluster; no cluster lifecycle."""

import json
import os
from pathlib import Path
import shlex
import subprocess

from .common import write_json


def render_script(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    runtime, resource = plan["runtime"], plan["resources"]
    address = plan.get("ray", {}).get("address")
    if address in (None, "auto"):
        address = os.environ.get("RAY_ADDRESS")
    if not address or address == "auto" or "$" in address:
        raise ValueError("Slurm driver needs the existing cluster address in ray.address or RAY_ADDRESS")
    directives = {"job-name": "evenet-" + plan["name"], "nodes": 1, "ntasks": 1,
                  "cpus-per-task": resource.get("driver_cpus", 2), "time": resource.get("time", "02:00:00"),
                  "output": str(plan_path.parent / "slurm-%j.log")}
    directives.update({key: resource[key] for key in ("account", "qos", "partition", "constraint") if resource.get(key)})
    if runtime.get("image"):
        directives["image"] = runtime["image"]
    lines = ["#!/bin/bash"]
    for key, value in directives.items():
        if any(c in str(value) for c in "\r\n") or "$" in str(value):
            raise ValueError(f"Invalid or unresolved Slurm directive: {key}")
        lines.append(f"#SBATCH --{key}={shlex.quote(str(value))}")
    lines += ["set -euo pipefail", "export PYTHONUNBUFFERED=1"]
    lines.append(f"export RAY_ADDRESS={shlex.quote(address)}")
    root = str(Path(__file__).resolve().parents[1])
    lines.append(f"export PYTHONPATH={shlex.quote(root)}${{PYTHONPATH:+:$PYTHONPATH}}")
    prefix = ["shifter", *runtime.get("shifter_args", [])] if runtime.get("image") else []
    command = ["srun", "--nodes=1", "--ntasks=1", "--cpu-bind=none", *prefix,
               *runtime.get("python", ["python3"]), "-m", "benchmark.workflow", "_execute", str(plan_path)]
    lines.append("exec " + shlex.join(command))
    return "\n".join(lines) + "\n"


def submit(plan_path, dry_run=False):
    plan_path = Path(plan_path)
    script = plan_path.parent / "driver.sbatch"
    script.write_text(render_script(plan_path))
    print(script)
    if dry_run:
        return
    result = subprocess.check_output(["sbatch", "--parsable", str(script)], text=True).strip()
    job_id = result.split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unrecognized sbatch response: {result}")
    write_json(plan_path.parent / "submission.json", {"job_id": job_id, "script": str(script), "response": result})
    print(f"Submitted driver job {job_id}. Ray cluster remains user-managed.")
    print(f"Progress: evenet-benchmark status {shlex.quote(str(plan_path))} --watch")
    print(f"Cancel this driver only: scancel {job_id}")
