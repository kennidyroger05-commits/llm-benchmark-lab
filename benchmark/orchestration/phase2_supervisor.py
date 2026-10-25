"""Process boundary for CUDA lifetime and disposable task state. Linux only."""
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal
import time
import uuid

from benchmark.tools.sandbox import DockerSandboxBackend


class ScopedDockerBackend(DockerSandboxBackend):
    """Same validated Docker isolation, plus ownership label for crash cleanup."""
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
    def _create_args(self, name, script):
        args = super()._create_args(name, script)
        # Options precede the image (inserting immediately after `create`).
        return args[:1] + ['--label', 'phase2.owner=' + self.owner] + args[1:]


def _entry(connection, target, settings, directory, owner):
    os.setsid()
    try:
        handler = target(settings, owner)
        connection.send('ready')
        while True:
            command = connection.recv()
            if command is None:
                break
            result = handler(command)
            # No large pipe writes: parent waits on a tiny completion signal.
            path = Path(directory) / 'result.pickle'
            with path.open('wb') as stream:
                pickle.dump(result, stream)
            connection.send('done')
    except BaseException:
        try:
            connection.send('worker_failure')
        except Exception:
            pass
    finally:
        connection.close()


class ProcessSupervisor:
    """One loaded worker, sequential commands. A killed CUDA worker ends the run.

    Never fork an initialized CUDA context. Spawn starts a new session; on expiry
    kill its process group, join, then remove only containers with our random
    ownership label. Cleanup failure is fatal, never followed by another task.
    """
    def __init__(self, target, settings, directory, *, startup_seconds=1800):
        self.owner = uuid.uuid4().hex
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        parent, child = mp.get_context('spawn').Pipe()
        self.connection = parent
        self.process = mp.get_context('spawn').Process(target=_entry,
            args=(child, target, settings, str(directory), self.owner))
        self.process.start()
        child.close()
        if not parent.poll(startup_seconds) or parent.recv() != 'ready':
            self.close()
            raise RuntimeError('worker_startup_failure')

    def execute(self, command, *, timeout=600):
        if not self.process.is_alive():
            raise RuntimeError('worker_not_alive')
        start = time.monotonic()
        self.connection.send(command)
        if not self.connection.poll(max(0, timeout - (time.monotonic() - start))):
            self.close()
            return {'supervision': 'timeout', 'technical_success': False}
        try:
            signal_value = self.connection.recv()
        except EOFError:
            signal_value = 'worker_failure'
        if signal_value != 'done':
            self.close()
            return {'supervision': 'infrastructure_failure', 'technical_success': False}
        # Trusted worker-created local pickle only, never model-supplied data.
        with (self.directory / 'result.pickle').open('rb') as stream:
            return {'supervision': 'completed', 'result': pickle.load(stream)}

    def close(self):
        descendants = {}
        creation_unresolved = False
        def info(pid):
            try:
                fields = Path('/proc/' + str(pid) + '/stat').read_text().split(') ', 1)[1].split()
                return fields[0], int(fields[1]), fields[19]
            except (OSError, IndexError, ValueError):
                return None
        if self.process.is_alive():
            # Docker attach clients deliberately start their own session. Freeze
            # the worker before discovering descendants, so it cannot spawn more.
            try:
                os.kill(self.process.pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
            candidates = {int(p.name): info(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
            parents = {self.process.pid}
            while True:
                found = {pid for pid, data in candidates.items() if data and data[1] in parents} - parents
                if not found:
                    break
                parents |= found
            descendants = {pid: candidates[pid][2] for pid in parents - {self.process.pid}}
            for pid, started in descendants.items():
                data = info(pid)
                if data and data[2] == started:
                    try:
                        os.kill(pid, signal.SIGSTOP)
                    except ProcessLookupError:
                        pass
            # A Docker create request can outlive a killed CLI in the daemon.
            # Keep the producer stopped and let in-flight create clients finish
            # before killing the group and querying ownership labels. Attach
            # clients are killed below; they need not finish naturally.
            creation_deadline = time.monotonic() + 10
            creation_unresolved = False
            while True:
                creating = []
                for pid, started in descendants.items():
                    data = info(pid)
                    if not data or data[2] != started or data[0] == 'Z':
                        continue
                    try:
                        args = Path('/proc/' + str(pid) + '/cmdline').read_bytes().split(b'\0')
                    except OSError:
                        continue
                    if b'create' in args and ('phase2.owner=' + self.owner).encode() in args:
                        creating.append(pid)
                        try:
                            os.kill(pid, signal.SIGCONT)
                        except ProcessLookupError:
                            pass
                if not creating:
                    break
                if time.monotonic() >= creation_deadline:
                    creation_unresolved = True
                    break
                time.sleep(.02)
            for pid, started in descendants.items():
                data = info(pid)
                if data and data[2] == started:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                self.process.kill()  # Worker may not yet have reached setsid.
        self.process.join(timeout=10)
        if self.process.is_alive():
            raise RuntimeError('worker_cleanup_failed')
        backend = DockerSandboxBackend()
        if backend.docker:
            ids = backend._docker(['ps', '-aq', '--filter', 'label=phase2.owner=' + self.owner])
            if ids.returncode:
                raise RuntimeError('container_cleanup_failed')
            for identifier in ids.stdout.decode().split():
                if backend._docker(['rm', '-f', identifier]).returncode:
                    raise RuntimeError('container_cleanup_failed')
            verify = backend._docker(['ps', '-aq', '--filter', 'label=phase2.owner=' + self.owner])
            if verify.returncode or verify.stdout.strip():
                raise RuntimeError('container_cleanup_failed')
        deadline = time.monotonic() + 10
        while any((data := info(pid)) and data[2] == started and data[0] != 'Z'
                  for pid, started in descendants.items()):
            if time.monotonic() >= deadline:
                raise RuntimeError('descendant_cleanup_failed')
            time.sleep(.02)
        self.connection.close()
        if creation_unresolved:
            raise RuntimeError('container_creation_unresolved')
