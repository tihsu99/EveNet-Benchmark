"""Small shared helpers; no EveNet imports in the orchestration process."""

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import tempfile

import yaml
from tqdm.auto import tqdm


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        except BaseException:
            os.unlink(stream.name)
            raise
    os.replace(stream.name, path)


def read_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream) or {}


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def path_value(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("A nonempty configured path is required")
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded or "<" in expanded or "[your" in expanded:
        raise ValueError(f"Unresolved path: {value}")
    path = Path(expanded)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def load_suite(path):
    path = Path(path).resolve()
    suite = read_yaml(path)
    if suite.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    datasets = {}
    for name, task in suite.get("benchmarks", {}).items():
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError(f"Unsafe benchmark name: {name}")
        group = task.get("data_group", name)
        if not isinstance(group, str) or not group or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in group):
            raise ValueError(f"Unsafe data group: {group}")
        if task.get("enabled", True):
            if group in datasets and datasets[group] != task.get("data"):
                raise ValueError(f"Shared data group has different selections: {group}")
            datasets[group] = task.get("data")
    return suite, path.parent


def replace_tokens(value, tokens):
    if isinstance(value, dict):
        return {key: replace_tokens(item, tokens) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_tokens(item, tokens) for item in value]
    if isinstance(value, str):
        for key, item in tokens.items():
            value = value.replace(key, str(item))
    return value


def resolve_template(path, repository=None, tokens=None):
    """Resolve EveNet section defaults, including public examples rooted at the repo."""
    path = Path(path)
    result = replace_tokens(read_yaml(path), tokens or {})
    for key, value in list(result.items()):
        if isinstance(value, dict) and "default" in value:
            override = dict(value)
            relative = override.pop("default")
            source = path_value(relative, path.parent)
            if not source.exists() and repository:
                source = path_value(relative, repository)
            result[key] = merge(replace_tokens(read_yaml(source), tokens or {}), override)
    return result


def source_revision(path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "diff_sha256": digest(git("diff", "HEAD")),
                "submodules": git("submodule", "status", "--recursive")}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None}


def run_command(argv, cwd, log, env=None, cancel_event=None):
    """Use argv, never a shell; preserve full command logs for failed runs."""
    if not argv or not all(isinstance(value, str) for value in argv):
        raise ValueError("Commands must be nonempty lists of strings")
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as stream:
        stream.write(json.dumps({"argv": argv, "cwd": str(cwd)}) + "\n")
        stream.flush()
        stop = threading.Event()
        started = time.monotonic()
        process = None

        def terminate():
            if process is not None and process.poll() is None:
                try:
                    if cancel_event is not None:
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if cancel_event is not None:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass

        def heartbeat():
            next_report = started + 30
            while not stop.wait(1):
                if cancel_event is not None and cancel_event.is_set():
                    terminate()
                    return
                if time.monotonic() >= next_report:
                    tqdm.write(f"[{Path(log).stem}] running {time.monotonic() - started:.0f}s | log: {log}")
                    next_report = time.monotonic() + 30

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        environment = dict(os.environ if env is None else env)
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Probe execution cancelled")
            with subprocess.Popen(argv, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                                  start_new_session=cancel_event is not None) as process:
                try:
                    for line in process.stdout:
                        stream.write(line)
                        stream.flush()
                        tqdm.write(line.rstrip())
                    if process.wait():
                        raise subprocess.CalledProcessError(process.returncode, argv)
                except BaseException:
                    terminate()
                    raise
        finally:
            stop.set()
            thread.join()
