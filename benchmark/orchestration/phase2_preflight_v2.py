"""Protocol V2 blocking preflight: AST self-tests replace Docker, Python 3.13 allowed."""
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import sys

from .phase2_preflight import ROOT, sha, slug, _cuda, _fp16, _auth, _model_checks, _writable, BENCHMARK_HASH
from .revision_v2 import REGISTRY
from benchmark.tools.catalog_v2 import public_catalog_sha256
from benchmark.tools.restricted_python import RestrictedASTBackend

FROZEN = ROOT / 'benchmark/phase2_frozen_v2.json'
# Filled from the explicit reviewed public projection, not from model output.
CATALOG_HASH = 'f802970524fa3daadb8e30801aa40ed81995b1b8d0f1fd64053e34b178ddc112'


def frozen_integrity():
    return all((ROOT / name).is_file() and sha(ROOT / name) == digest
               for name, digest in json.loads(FROZEN.read_text()).items())


def compatible_python(info=None):
    return (3, 10) <= (sys.version_info[:2] if info is None else info) < (3, 14)


def _versions():
    # A compatibility envelope, not an assertion that Colab's preinstalled
    # versions equal those used in Phase 1. Exact installed versions go in manifests.
    minima = {'torch': (2, 8), 'transformers': (4, 56), 'accelerate': (1, 10)}
    maxima = {'torch': 3, 'transformers': 5, 'accelerate': 2}
    for package, lower in minima.items():
        release = tuple(int(p) for p in version(package).split('+')[0].split('.')[:2])
        if release < lower or release[0] >= maxima[package]: return False
    return True


def run_preflight(model_id, output_root, *, inspect_model=False, probes=None):
    probes = {} if probes is None else probes
    checks = []
    def check(name, fn):
        try: passed = bool(probes.get(name, fn)())
        except Exception: passed = False
        checks.append({'requirement': name, 'status': 'PASS' if passed else 'FAIL',
                       'code': None if passed else name + '_failed'})
    from benchmark.providers.transformers_phase2 import OFFICIAL_MODELS, GENERATION_OPTIONS
    check('official_model', lambda: model_id in OFFICIAL_MODELS)
    check('python', lambda: sys.platform == 'linux' and compatible_python())
    for name in ('torch', 'transformers', 'accelerate'):
        check(name, lambda n=name: import_module(n) is not None)
    check('dependency_versions', _versions)
    check('cuda_gpu', _cuda); check('float16', _fp16); check('hf_auth', _auth)
    check('catalog_hash', lambda: public_catalog_sha256() == CATALOG_HASH)
    check('benchmark_hash', lambda: sha(ROOT / 'prompts/benchmark_v1.json') == BENCHMARK_HASH)
    check('frozen_files', frozen_integrity)
    check('registry', lambda: len(REGISTRY) == 25)
    check('restricted_python_runtime', lambda: RestrictedASTBackend().validate_environment())
    check('no_unsafe_fallback', lambda: dict(GENERATION_OPTIONS) == {
        'max_new_tokens': 512, 'do_sample': False, 'temperature': None, 'seed': None})
    output = Path(output_root)
    check('output_writable', lambda: _writable(output))
    check('no_conflicting_run', lambda: not any((output / 'results/raw/phase2' / slug(model_id)).glob('benchmark_*.json')))
    names = ('model_access', 'tokenizer', 'model_config', 'chat_template', 'context_capacity')
    inspected = {}
    if inspect_model:
        try: inspected = probes['model_checks']() if 'model_checks' in probes else _model_checks(model_id)
        except Exception: pass
        for name in names: check(name, lambda n=name: inspected.get(n, False))
    else:
        checks.extend({'requirement': name, 'status': 'not_executed_until_colab',
                       'code': 'not_executed_until_colab'} for name in names)
    return {'status': 'PASS' if all(c['status'] == 'PASS' for c in checks) else 'FAIL',
            'checks': checks, 'model_id': model_id, 'model_revision': inspected.get('revision'),
            'protocol_version': 'phase2_v2', 'sandbox_mode': 'restricted_ast_interpreter',
            'catalog_hash': public_catalog_sha256()}
