"""Deterministic Phase 2 reports; public projections exclude answers and fixtures."""
from copy import deepcopy
import hashlib
from pathlib import Path

from benchmark.orchestration.tool_trace import finite_json_bytes
from benchmark.tools.catalog import public_catalog_sha256
from .phase2 import EVALUATION_VERSION
from .phase2_efficiency import POLICY_VERSION
from .phase2_aggregation import aggregate_run

_ROOT = Path(__file__).resolve().parents[2]
# Hash evaluator code and execution/contract dependencies, not only the entrypoint.
_HASH_SOURCES = tuple(sorted(
    [p.relative_to(_ROOT).as_posix() for p in Path(__file__).parent.glob('*.py')]
    + ['benchmark/orchestration/' + name + '.py' for name in (
        'protocol', 'dispatcher', 'tool_trace', 'episode')]
    + ['benchmark/tools/' + name for name in (
        'catalog.py', '_schema_definitions.py', 'schemas.py', 'domain_tools.py',
        'sandbox.py', '_sandbox_worker.py')]))
# Reports expose only metric fields, never the whole EpisodeResult/task record.
_TASK_FIELDS = frozenset('''evaluation_version task_id category model_id run_id
technical_success protocol_correct task_correct strict_correct content_correct
format_correct dataset_tools_contract_correct direct_answer termination_reason
infrastructure_failure domain_actions_correct evaluation_error tool_calls_total
tool_calls_dispatched tool_calls_successful tool_calls_failed duplicates invalid
switches sequence_correct unnecessary_tool_calls minimum_required calls_over_minimum
calls_below_minimum selection_valid tool_selection_class tool_calls_blocked'''.split())
_RUN_FIELDS = {'run_id', 'attempt_id', 'model_revision', 'tokenizer_revision', 'python_version',
               'torch_version', 'transformers_version', 'accelerate_version'}
_HARDWARE_FIELDS = {'device', 'gpu_name', 'vram_gb', 'dtype', 'device_map', 'offload'}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _metadata(value, allowed):
    data = {} if value is None else value
    if not isinstance(data, dict) or not set(data) <= allowed:
        raise ValueError('Unsupported report metadata fields.')
    # Metadata is deliberately small: nested controller/task records are not
    # valid version strings or hardware descriptors, even under an allowed key.
    for key, item in data.items():
        if item is None:
            continue
        if allowed == _RUN_FIELDS:
            valid = type(item) is str
        elif key == 'device_map':
            valid = type(item) is str or (type(item) is dict and all(
                type(k) is str and type(v) in (str, int) for k, v in item.items()))
        elif key == 'offload':
            valid = type(item) is bool
        elif key == 'vram_gb':
            valid = type(item) in (int, float) and item >= 0
        else:
            valid = type(item) is str
        if not valid:
            raise ValueError('Unsupported report metadata value.')
    finite_json_bytes(data)
    return deepcopy(data)


def build_report(evaluations, *, model_id, run_metadata=None, hardware_metadata=None,
                 benchmark_path=None, source_raw_path=None, source_raw_hash=None):
    """Explicit metadata only; no clock, random ID, private task payload or trace.

    Only metric keys are projected from evaluations. Inputs must belong to one
    run/model; source paths/hashes are provenance, never raw contents. Benchmark
    bytes are hashed as stored, with no normalization.
    """
    run = _metadata(run_metadata, _RUN_FIELDS)
    hardware = _metadata(hardware_metadata, _HARDWARE_FIELDS)
    rows = [{key: deepcopy(row[key]) for key in sorted(_TASK_FIELDS)} for row in evaluations]
    rows.sort(key=lambda row: int(row['task_id'][1:]))
    if 'run_id' not in run:
        run_ids = {row['run_id'] for row in rows}
        if len(run_ids) == 1:
            run['run_id'] = next(iter(run_ids))
    summary = aggregate_run(rows, model_id=model_id, run_id=run.get('run_id'))
    source = None
    if source_raw_path is not None:
        digest = _sha(Path(source_raw_path).read_bytes())
        if source_raw_hash is not None and source_raw_hash != digest:
            raise ValueError('Source hash mismatch.')
        source = {'path': str(source_raw_path), 'sha256': digest}
    elif source_raw_hash is not None:
        if (not isinstance(source_raw_hash, str) or len(source_raw_hash) != 64
                or any(c not in '0123456789abcdef' for c in source_raw_hash)):
            raise ValueError('Invalid source hash.')
        source = {'path': None, 'sha256': source_raw_hash}
    path = _ROOT / 'prompts/benchmark_v1.json' if benchmark_path is None else Path(benchmark_path)
    report = {
        'evaluation_version': EVALUATION_VERSION, 'efficiency_policy_version': POLICY_VERSION,
        'phase': 'phase2', 'condition': 'tools', 'model_id': model_id,
        'benchmark_hash': _sha(path.read_bytes()), 'catalog_hash': public_catalog_sha256(),
        'evaluator_hashes': {name: _sha((_ROOT / name).read_bytes()) for name in _HASH_SOURCES},
        'run_metadata': run, 'hardware_metadata': hardware, 'source_raw': source,
        'task_evaluations': rows, 'summary': summary,
        'tool_metrics': {key: value for key, value in summary.items()
                         if key.startswith(('tool_calls_', 'selection_class_', 'calls_over_'))
                         or key in {'duplicates', 'invalid', 'switches', 'unnecessary_tool_calls'}},
    }
    report['structural_hash'] = _sha(finite_json_bytes(report))
    return report


def report_bytes(report):
    """Canonical sorted UTF-8 JSON plus LF. Verify integrity before persistence."""
    content = {key: value for key, value in report.items() if key != 'structural_hash'}
    if _sha(finite_json_bytes(content)) != report.get('structural_hash'):
        raise ValueError('Report structural hash mismatch.')
    return finite_json_bytes(report) + b'\n'


def export_report(report, destination):
    """Write only a new file; never overwrite Phase 1, RAW or existing reports.

    Caller selects a path, normally reports/phase2/<model>/<run>.json. Outside
    the project (e.g. a temporary test directory) no project files are touched.
    """
    path = Path(destination)
    resolved = path.resolve()
    if resolved.is_relative_to(_ROOT) and not any(resolved.is_relative_to(_ROOT / prefix)
            for prefix in ('reports/phase2', 'results/processed/phase2')):
        raise ValueError('Phase 2 report destination must be separate from protected artifacts.')
    data = report_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
    return path
