"""Real Docker integration plus infrastructure classification; no model execution."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from benchmark.tools.sandbox import (
    DockerSandboxBackend, SandboxBackend, SandboxError, validate_phase2_environment,
)


class SandboxUnitTests(unittest.TestCase):
    def test_no_docker_is_unavailable(self):
        backend=DockerSandboxBackend()
        backend.docker=None
        self.assertFalse(backend.is_available())
        with self.assertRaises(SandboxError) as caught: backend.validate_environment()
        self.assertEqual(caught.exception.origin,"backend_infrastructure")

    def test_bad_image_no_dynamic_pull(self):
        backend=DockerSandboxBackend(image="phase2-no-such-image:never")
        self.assertFalse(backend.is_available())
        self.assertIn("--pull=never",backend._create_args("example","print(1)"))

    def test_backend_failure_distinct(self):
        backend=DockerSandboxBackend()
        with patch.object(backend,"_docker",side_effect=SandboxError("execution_error","backend_infrastructure")):
            self.assertFalse(backend.is_available())

    def test_command_has_no_mounts_or_host_environment(self):
        args=DockerSandboxBackend()._create_args("example","print(1)")
        for flag in ["--network","--read-only","--memory","--pids-limit","--cpus",
                     "--cap-drop","--user","--security-opt"]:
            self.assertIn(flag,args)
        for flag in ["--privileged","--volume","-v","--mount","--env-file"]:
            self.assertNotIn(flag,args)


class DockerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend=DockerSandboxBackend()
        if not cls.backend.is_available():
            raise AssertionError("Docker integration required, never silently skipped.")

    def probe(self,script,**kwargs):
        stdout,stderr=self.backend._launch(script,**kwargs)
        self.assertEqual(stderr,b"")
        return json.loads(stdout)

    def test_available_and_full_preflight(self):
        result=validate_phase2_environment(self.backend)
        self.assertTrue(result["timeout_verified"])
        self.assertTrue(result["network_blocked"])
        self.assertTrue(result["repo_hidden"])
        self.assertEqual(result["memory_bytes"],268435456)
        self.assertRegex(result["runtime"],r"^3\.12\.")

    def test_real_runtime_resource_configuration(self):
        result=self.probe(
            "import json,os\nfrom pathlib import Path\n"
            "print(json.dumps({'uid':os.getuid(),'memory':Path('/sys/fs/cgroup/memory.max').read_text().strip(),"
            "'pids':Path('/sys/fs/cgroup/pids.max').read_text().strip(),"
            "'cpu':Path('/sys/fs/cgroup/cpu.max').read_text().strip()}))")
        self.assertNotEqual(result["uid"],0)
        self.assertEqual(result["memory"],"268435456")
        self.assertEqual(result["pids"],"16")
        quota,period=map(int,result["cpu"].split())
        self.assertEqual(quota,period)

    def test_host_repo_socket_and_secret_invisible(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/"private-host-marker"
            marker.write_text("not-for-container")
            script=("import json,os\nprint(json.dumps({'host':os.path.exists("+repr(str(marker))+"),"
                    "'repo':os.path.exists('/workspace/llm-benchmark-lab'),"
                    "'socket':os.path.exists('/var/run/docker.sock'),"
                    "'secret':os.environ.get('PHASE2_FAKE_SECRET'),"
                    "'hf':os.environ.get('HF_TOKEN'),'github':os.environ.get('GITHUB_TOKEN')}))")
            with patch.dict(os.environ,{"PHASE2_FAKE_SECRET":"do-not-forward","HF_TOKEN":"fake","GITHUB_TOKEN":"fake"}):
                result=self.probe(script)
            self.assertEqual(result,{"host":False,"repo":False,"socket":False,
                                     "secret":None,"hf":None,"github":None})

    def test_root_readonly_tmp_writable(self):
        result=self.probe(
            "import json\nout={}\n"
            "try:\n open('/write-here','w').close(); out['root']=True\n"
            "except OSError:\n out['root']=False\n"
            "with open('/tmp/write-here','w') as f: f.write('ok')\n"
            "out['tmp']=open('/tmp/write-here').read()\nprint(json.dumps(out))")
        self.assertEqual(result,{"root":False,"tmp":"ok"})

    def test_network_unreachable(self):
        result=self.probe(
            "import socket,json\ns=socket.socket(); s.settimeout(.1)\n"
            "try:\n s.connect(('198.51.100.1',80)); connected=True\n"
            "except OSError:\n connected=False\n"
            "finally:\n s.close()\nprint(json.dumps(connected))")
        self.assertFalse(result)

    def test_memory_limit_submitted_resource_failure(self):
        with self.assertRaises(SandboxError) as caught:
            self.backend._launch("x=bytearray(600*1024*1024)\nprint(len(x))",timeout=2)
        self.assertEqual(caught.exception.code,"resource_limit")
        self.assertEqual(caught.exception.origin,"submitted_code")

    def test_process_limit_enforced(self):
        result=self.probe(
            "import os,signal,json,time\nchildren=[]\n"
            "try:\n"
            " for _ in range(40):\n"
            "  pid=os.fork()\n"
            "  if pid==0:\n"
            "   time.sleep(10); os._exit(0)\n"
            "  children.append(pid)\n"
            "except OSError:\n limited=True\n"
            "else:\n limited=False\n"
            "finally:\n"
            " for pid in children: os.kill(pid,signal.SIGKILL)\n"
            " for pid in children: os.waitpid(pid,0)\n"
            "print(json.dumps({'limited':limited,'count':len(children)}))")
        self.assertTrue(result["limited"])
        self.assertLess(result["count"],16)

    def test_timeout_and_container_cleanup(self):
        created=[]
        original=self.backend._create_args
        def capture(name,script):
            created.append(name)
            return original(name,script)
        with patch.object(self.backend,"_create_args",side_effect=capture):
            with self.assertRaises(SandboxError) as caught:
                self.backend._launch("while True: pass",timeout=1)
        self.assertEqual(caught.exception.code,"tool_timeout")
        self.assertEqual(caught.exception.origin,"submitted_code")
        self.assertTrue(created)
        for name in created:
            result=self.backend._docker(["container","inspect",name])
            self.assertNotEqual(result.returncode,0)

    def test_cleanup_after_success_and_worker_error(self):
        names=[]
        original=self.backend._create_args
        def capture(name,script):
            names.append(name)
            return original(name,script)
        with patch.object(self.backend,"_create_args",side_effect=capture):
            self.probe("import json\nprint(json.dumps(1))")
            with self.assertRaises(SandboxError):
                self.backend.run_python(code="def main(data): raise ValueError('bad')",
                                        function="main",mode="executor")
        self.assertTrue(all(self.backend._docker(["inspect",n]).returncode!=0 for n in names))

    def test_explicit_cancel_removes_live_container(self):
        names=[]; errors=[]
        original=self.backend._create_args
        def capture(name,script):
            names.append(name); return original(name,script)
        def run():
            try: self.backend._launch("while True: pass",timeout=5)
            except SandboxError as error: errors.append(error)
        with patch.object(self.backend,"_create_args",side_effect=capture):
            thread=threading.Thread(target=run)
            thread.start()
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                if names:
                    state=self.backend._docker(["inspect","--format","{{.State.Running}}",names[0]])
                    if state.returncode==0 and state.stdout.strip()==b"true": break
                time.sleep(.02)
            self.backend.cancel()
            thread.join(timeout=4)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertEqual(errors[0].code,"tool_timeout")
        self.assertNotEqual(self.backend._docker(["inspect",names[0]]).returncode,0)
