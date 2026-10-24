"""Fail-closed Docker backend. Never starts a local Python subprocess for model code."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from threading import RLock
import time
import uuid

from .math_tools import ToolExecutionError
from ._sandbox_worker import PolicyError, json_bytes, pack, unpack, validate_code

# Preinstalled image, pinned by content digest. Calls use --pull=never.
DEFAULT_IMAGE = "python@sha256:b658b05ce9c4eb97ec6fc19424818a1f31993659c2b505a532f470e83807815d"
PROBE = """
import json,os,socket
checks = {}
checks["non_root"] = os.getuid() != 0
checks["repo_hidden"] = not os.path.exists("/workspace/llm-benchmark-lab")
checks["socket_hidden"] = not os.path.exists("/var/run/docker.sock")
checks["env_clean"] = all(k not in os.environ for k in ("HF_TOKEN","GITHUB_TOKEN","PHASE2_FAKE_SECRET"))
try:
 open("/forbidden","w").close(); checks["read_only"] = False
except OSError: checks["read_only"] = True
with open("/tmp/probe","w") as f: f.write("ok")
checks["tmp_writable"] = True
s = socket.socket(); s.settimeout(0.15)
try:
 s.connect(("198.51.100.1",80)); checks["network_blocked"] = False
except OSError: checks["network_blocked"] = True
finally: s.close()
from pathlib import Path
try:
 checks["memory_limited"] = Path("/sys/fs/cgroup/memory.max").read_text().strip() == "268435456"
 checks["pids_limited"] = Path("/sys/fs/cgroup/pids.max").read_text().strip() == "16"
 quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
 checks["cpu_limited"] = quota != "max" and int(quota) <= int(period)
except (OSError, ValueError):
 checks["memory_limited"] = checks["pids_limited"] = checks["cpu_limited"] = False
checks["runtime"] = __import__("sys").version.split()[0]
print(json.dumps(checks))
"""


class SandboxError(ToolExecutionError):
    """Origin/scope are controller diagnostics, never embedded in public messages."""
    def __init__(self, code, origin, scope="tool"):
        super().__init__(code, "Sandbox execution failed." if code == "execution_error"
                         else "Submitted code could not complete.")
        self.origin = origin
        self.scope = scope


@dataclass(frozen=True)
class SandboxResult:
    result: dict
    stdout: str = ""
    stderr: str = ""


class SandboxBackend(ABC):
    @abstractmethod
    def is_available(self): ...
    @abstractmethod
    def validate_environment(self): ...
    @abstractmethod
    def run_python(self, *, code, function, mode, data=None, cases=None, timeout=2): ...
    @abstractmethod
    def cancel(self): ...
    @abstractmethod
    def cleanup(self): ...


class DockerSandboxBackend(SandboxBackend):
    """One non-root, read-only container per execution. Cleanup failures are infrastructure."""
    def __init__(self, image=DEFAULT_IMAGE):
        self.image = image
        self.docker = shutil.which("docker")
        self.__active = set()
        self.__cancelled = set()
        self.__lock = RLock()

    def _docker(self, arguments, timeout=5, **kwargs):
        if not self.docker:
            raise SandboxError("execution_error", "backend_infrastructure")
        try:
            return subprocess.run([self.docker, *arguments], capture_output=True,
                                  timeout=timeout, env={"PATH": os.defpath}, **kwargs)
        except (OSError, subprocess.TimeoutExpired):
            raise SandboxError("execution_error", "backend_infrastructure") from None

    def is_available(self):
        if not self.docker:
            return False
        try:
            return (self._docker(["info"], timeout=3).returncode == 0 and
                    self._docker(["image", "inspect", self.image], timeout=3).returncode == 0)
        except SandboxError:
            return False

    def _create_args(self, name, script):
        return [
            "create", "--pull=never", "--name", name, "--label", "llm-benchmark-lab.sandbox=phase2",
            "--interactive", "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=8m,mode=1777",
            "--memory", "256m", "--memory-swap", "256m", "--pids-limit", "16",
            "--cpus", "1", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534", "--workdir", "/tmp",
            "--env", "PYTHONHASHSEED=0", "--env", "PYTHONIOENCODING=utf-8", "--env", "HOME=/tmp",
            "--entrypoint", "python", self.image, "-I", "-S", "-c", script,
        ]

    def _remove(self, name):
        # cancel() and the execution's finally block may race; serialize removal
        # so neither mistakes the other's in-progress cleanup for a failure.
        with self.__lock:
            deadline = time.monotonic() + 3
            while True:
                result = self._docker(["rm", "--force", name])
                check = self._docker(["container", "inspect", name])
                absent = any(marker in check.stderr for marker in
                             (b"No such container", b"No such object"))
                if check.returncode != 0 and (result.returncode == 0 or absent):
                    break
                if time.monotonic() >= deadline:
                    raise SandboxError("execution_error", "backend_infrastructure", "cleanup")
                time.sleep(.02)
            self.__active.discard(name)

    def _launch(self, script, payload=b"", timeout=2):
        """Trusted controller path; public run_python always applies the AST policy."""
        deadline = time.monotonic() + timeout
        name = "llm-phase2-" + uuid.uuid4().hex
        with self.__lock:
            self.__active.add(name)
        process = None
        try:
            creation = self._docker(self._create_args(name, script), timeout=5)
            if creation.returncode:
                raise SandboxError("execution_error", "backend_infrastructure")
            with self.__lock:
                if name in self.__cancelled:
                    raise SandboxError("tool_timeout", "submitted_code", "cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SandboxError("tool_timeout", "submitted_code")
            try:
                process = subprocess.Popen(
                    [self.docker, "start", "--attach", "--interactive", name],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env={"PATH": os.defpath}, start_new_session=True,
                )
                stdout, stderr = process.communicate(payload, timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                self._docker(["kill", name])
                # A container which exited at the deadline is also safe to remove.
                status = self._docker(["inspect", "--format", "{{.State.Running}}", name])
                if status.returncode or status.stdout.strip() != b"false":
                    raise SandboxError("execution_error", "backend_infrastructure", "cancellation")
                raise SandboxError("tool_timeout", "submitted_code") from None
            except OSError:
                raise SandboxError("execution_error", "backend_infrastructure") from None
            with self.__lock:
                cancelled = name in self.__cancelled
            if cancelled:
                raise SandboxError("tool_timeout", "submitted_code", "cancelled")
            if process.returncode:
                state = self._docker(["inspect", "--format", "{{.State.OOMKilled}}", name])
                if (state.returncode == 0 and state.stdout.strip() == b"true") or process.returncode in (137, 139):
                    raise SandboxError("resource_limit", "submitted_code")
                raise SandboxError("execution_error", "backend_infrastructure")
            return stdout, stderr
        finally:
            try:
                self._remove(name)
            finally:
                with self.__lock:
                    self.__cancelled.discard(name)

    def cancel(self):
        with self.__lock:
            names = tuple(self.__active)
            self.__cancelled.update(names)
        for name in names:
            self._remove(name)

    def cleanup(self):
        self.cancel()

    def validate_environment(self):
        if not self.is_available():
            raise SandboxError("execution_error", "backend_infrastructure", "preflight")
        stdout, _ = self._launch(PROBE, timeout=5)
        try:
            checks = json.loads(stdout)
            if not all(checks[k] for k in (
                    "non_root", "repo_hidden", "socket_hidden", "env_clean",
                    "read_only", "tmp_writable", "network_blocked",
                    "memory_limited", "pids_limited", "cpu_limited")):
                raise ValueError()
        except (ValueError, KeyError):
            raise SandboxError("execution_error", "backend_infrastructure", "preflight") from None
        # A live timeout probe proves forced cancellation and removal.
        try:
            self._launch("while True: pass", timeout=2)
        except SandboxError as error:
            if error.code != "tool_timeout" or error.origin != "submitted_code":
                raise
        else:
            raise SandboxError("execution_error", "backend_infrastructure", "preflight")
        return {**checks, "image": self.image, "memory_bytes": 256 * 1024**2,
                "pids_limit": 16, "cpu_limit": 1, "timeout_verified": True}

    def run_python(self, *, code, function, mode, data=None, cases=None, timeout=2):
        started = time.monotonic()
        try:
            validate_code(code, function, mode)
        except PolicyError:
            raise SandboxError("invalid_data", "sandbox_policy") from None
        # Creation with --pull=never checks daemon/image availability. Do not
        # spend the tool deadline querying Docker info/image on every call.
        if not self.docker:
            raise SandboxError("execution_error", "backend_infrastructure", "preflight")
        request = {"code": code, "function": function, "mode": mode,
                   "data": data, "cases": cases}
        try:
            payload = json.dumps(pack(request), ensure_ascii=True, allow_nan=False).encode()
        except (ValueError, TypeError, RecursionError):
            raise SandboxError("invalid_data", "submitted_code") from None
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise SandboxError("tool_timeout", "submitted_code")
        worker = Path(__file__).with_name("_sandbox_worker.py").read_text(encoding="utf-8")
        stdout, stderr = self._launch(worker, payload, timeout=remaining)
        try:
            reply = json.loads(stdout)
            if not reply["ok"]:
                raise SandboxError(reply["code"], reply["origin"])
            result = unpack(reply["result"])
            if stderr:
                raise SandboxError("execution_error", "backend_infrastructure")
            if len(json_bytes(result)) > 8192 or reply["stdout"] or reply["stderr"]:
                raise SandboxError("invalid_data", "submitted_code")
            return SandboxResult(result)
        except SandboxError:
            raise
        except (ValueError, KeyError, TypeError, RecursionError):
            raise SandboxError("execution_error", "backend_infrastructure") from None


def validate_phase2_environment(backend=None):
    """Future runners must call before loading a model; there is no unsafe fallback."""
    return (backend or DockerSandboxBackend()).validate_environment()
