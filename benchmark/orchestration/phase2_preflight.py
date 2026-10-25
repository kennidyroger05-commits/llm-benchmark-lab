"""Blocking PASS/FAIL checks. Environment-only inspection never accesses HF models."""
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile

from benchmark.tools.catalog import public_catalog_sha256, get_public_catalog
from benchmark.tools.sandbox import DockerSandboxBackend
from .dispatcher import TOOL_REGISTRY
from benchmark.providers.transformers_phase2 import OFFICIAL_MODELS

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'benchmark/phase2_frozen.json'
CATALOG_HASH = 'cfe666d04c4870461bd3c18bfd5b78a393d5778cd6fb2aac6399d6701dade92b'
BENCHMARK_HASH = 'ac96c5324a1eb84fe84fcd1178a99e1f7448104e9abb814dff8ed621bee085b4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slug(model_id):
    return model_id.replace('/', '--')


def frozen_integrity():
    files = json.loads(FROZEN.read_text())
    return all((ROOT / name).is_file() and sha(ROOT / name) == digest for name, digest in files.items())


def _cuda():
    import torch
    if not torch.cuda.is_available():
        return False
    index = torch.cuda.current_device()
    return bool(torch.cuda.get_device_name(index)) and torch.cuda.get_device_properties(index).total_memory > 0


def _fp16():
    import torch
    if not _cuda() or torch.cuda.get_device_capability() < (5, 3):
        return False
    # Exercise a small GPU FP16 operation, not a model.
    value = torch.ones(2, device='cuda', dtype=torch.float16)
    result = value + value
    torch.cuda.synchronize()
    return bool((result == 2).all().item())


def _versions():
    for line in (ROOT / 'constraints-phase2.txt').read_text().splitlines():
        if '==' in line:
            name, expected = line.split('==')
            if version(name) != expected:
                return False
    return True


def _safe_backend():
    from benchmark.tools.code_tools import run_python_executor, run_python_test_runner
    backend = DockerSandboxBackend()
    backend.validate_environment()
    assert run_python_executor({'code': 'def main(data): return data', 'data': 7}, backend)['value'] == 7
    result = run_python_test_runner({'code': 'def identity(data): return data', 'function': 'identity',
        'cases': [{'args': [7], 'kwargs': {}, 'expected': 7}]}, backend)
    return result['passed'] is True


def _auth():
    from huggingface_hub import get_token
    return bool(get_token())


def _model_checks(model_id):
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoTokenizer
    HfApi().auth_check(repo_id=model_id, repo_type='model', token=True)
    info = HfApi().model_info(model_id, token=True)
    revision = info.sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': 'pronto'}],
                                         tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    limits = [v for v in (getattr(config, 'max_position_embeddings', None),
              getattr(tokenizer, 'model_max_length', None)) if type(v) is int and 0 < v < 10**9]
    return {'model_access': True, 'tokenizer': True, 'model_config': True,
            'chat_template': bool(tokenizer.chat_template and text and encoded['input_ids']),
            'context_capacity': bool(limits and min(limits) >= 16384), 'revision': revision}


def _writable(path):
    if path.resolve().is_relative_to(ROOT) and path.resolve() != ROOT:
        return False
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path) as stream:
        stream.write(b'preflight')
    return True


def run_preflight(model_id, output_root, *, inspect_model=False, probes=None):
    """probes is a unit-test seam, never exposed by the production CLI.

    All mandatory checks must be PASS; deferred/failed probes cannot authorize a
    run. Messages are fixed codes, never exception strings/token values.
    """
    probes = {} if probes is None else probes
    checks = []
    def check(name, function, code=None):
        try:
            ok = bool(probes.get(name, function)())
        except Exception:
            ok = False
        checks.append({'requirement': name, 'status': 'PASS' if ok else 'FAIL',
                       'code': None if ok else (code or name + '_failed')})
    check('official_model', lambda: model_id in OFFICIAL_MODELS)
    check('python', lambda: sys.platform == 'linux' and (3, 10) <= sys.version_info[:2] < (3, 13))
    for package in ('torch', 'transformers', 'accelerate'):
        check(package, lambda name=package: importlib.import_module(name) is not None)
    check('dependency_versions', _versions)
    check('cuda_gpu', _cuda)
    check('float16', _fp16)
    check('hf_auth', _auth)
    check('catalog_hash', lambda: public_catalog_sha256() == CATALOG_HASH)
    check('benchmark_hash', lambda: sha(ROOT / 'prompts/benchmark_v1.json') == BENCHMARK_HASH)
    check('frozen_files', frozen_integrity)
    check('registry', lambda: len(TOOL_REGISTRY) == 25 and set(TOOL_REGISTRY) == {t.name for t in get_public_catalog()})
    check('safe_sandbox', _safe_backend, 'safe_sandbox_unavailable')
    from benchmark.providers.transformers_phase2 import GENERATION_OPTIONS, TransformersEpisodeProvider
    def configuration():
        core = TransformersEpisodeProvider(model_id)._core  # No import/load of weights.
        return (core.device_map == 'auto' and core.torch_dtype == 'float16'
                and dict(GENERATION_OPTIONS) == {'max_new_tokens': 512, 'do_sample': False,
                                                'temperature': None, 'seed': None})
    check('no_unsafe_fallback', configuration)
    output = Path(output_root)
    check('output_writable', lambda: _writable(output))
    check('no_conflicting_run', lambda: not any((output / 'results/raw/phase2' / slug(model_id)).glob('benchmark_*.json')))
    names = ('model_access', 'tokenizer', 'model_config', 'chat_template', 'context_capacity')
    revision = None
    if inspect_model:
        try:
            inspected = probes['model_checks']() if 'model_checks' in probes else _model_checks(model_id)
            revision = inspected.get('revision')
        except Exception:
            inspected = {}
        for name in names:
            check(name, lambda n=name: inspected.get(n, False))
    else:
        checks.extend({'requirement': name, 'status': 'not_executed_until_colab',
                       'code': 'not_executed_until_colab'} for name in names)
    return {'status': 'PASS' if all(c['status'] == 'PASS' for c in checks) else 'FAIL',
            'checks': checks, 'model_id': model_id, 'model_revision': revision}
