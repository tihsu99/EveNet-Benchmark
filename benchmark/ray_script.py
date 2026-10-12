"""Run a native non-Ray GPU script on a GPU reserved by the existing Ray cluster."""

import argparse
import os
import subprocess
import sys


def run_script(argv, cwd):
    # Ray sets CUDA_VISIBLE_DEVICES before this worker and its child start.
    subprocess.run(argv, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.cpus < 1 or args.gpus not in (0, 1):
        parser.error("Supply a command, positive CPUs, and 0 or 1 GPU")
    import ray
    env = {key: os.environ[key] for key in ("PYTHONPATH", "PYTHONHASHSEED", "WANDB_MODE")
           if key in os.environ}
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), runtime_env={"env_vars": env})
    try:
        task = ray.remote(num_cpus=args.cpus, num_gpus=args.gpus, max_retries=0)(run_script)
        ray.get(task.remote([sys.executable, *command], os.getcwd()))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
