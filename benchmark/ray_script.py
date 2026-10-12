"""Run a native non-Ray GPU script on a GPU reserved by the existing Ray cluster."""

import argparse
import os
import socket
import subprocess
import sys


def run_script(argv, cwd):
    # Ray sets CUDA_VISIBLE_DEVICES before this worker and its child start.
    subprocess.run(argv, cwd=cwd, check=True)


class ResourceLease:
    """Hold Ray resources while a host subprocess uses the assigned devices."""

    def gpu_ids(self):
        import ray
        return ray.get_gpu_ids()


def driver_node(ray, cpus, gpus):
    hostname = socket.gethostname()
    address = ray.util.get_node_ip_address()
    for node in ray.nodes():
        if node["Alive"] and (node.get("NodeManagerAddress") == address or
                              node.get("NodeManagerHostname") == hostname):
            capacity = node.get("Resources", {})
            if capacity.get("CPU", 0) < cpus or capacity.get("GPU", 0) < gpus:
                raise ValueError(f"Host native stages need {cpus} CPUs and {gpus} GPUs on the driver node")
            return node["NodeID"]
    raise ValueError("Host native stages require the driver to run on a node of the existing Ray cluster")


def run_on_driver(ray, command, cpus, gpus):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_id = driver_node(ray, cpus, gpus)
    actor = ray.remote(num_cpus=cpus, num_gpus=gpus,
                       scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False))(ResourceLease)
    lease = actor.remote()
    try:
        devices = ray.get(lease.gpu_ids.remote())
        if len(devices) != gpus:
            raise RuntimeError(f"Expected {gpus} reserved GPUs, got {devices}")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices))}
        subprocess.run(command, cwd=os.getcwd(), env=environment, check=True)
    finally:
        ray.kill(lease, no_restart=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--on-driver", action="store_true",
                        help="Run on the driver host while holding same-node Ray resources")
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
        if args.on_driver:
            run_on_driver(ray, [sys.executable, *command], args.cpus, args.gpus)
        else:
            task = ray.remote(num_cpus=args.cpus, num_gpus=args.gpus, max_retries=0)(run_script)
            ray.get(task.remote([sys.executable, *command], os.getcwd()))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
