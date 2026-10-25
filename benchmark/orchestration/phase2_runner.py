"""Official sequential Phase 2 coordinator, with injectable seams for smoke tests."""
from dataclasses import asdict
import hashlib
from importlib.metadata import version, PackageNotFoundError
import platform
import json
from pathlib import Path
import tempfile
import uuid

from benchmark.evaluation.phase2 import evaluate_episode
from benchmark.evaluation.phase2_export import build_report, export_report
from benchmark.orchestration.episode import EpisodeResult
from benchmark.orchestration.phase2_runtime import run_transformers_episode
from benchmark.providers.transformers_phase2 import TransformersEpisodeProvider
from benchmark.tools.domain_tools import DomainSession
from benchmark.tools._sandbox_worker import json_bytes
from .phase2_preflight import ROOT, FROZEN, run_preflight, sha, slug
from .phase2_supervisor import ProcessSupervisor, ScopedDockerBackend
from benchmark.providers.failure_diagnostics import failure_details


def official_worker(settings, owner):
    """Executed only in spawned worker; loading is never invoked by import."""
    from huggingface_hub import snapshot_download
    # Pin both weights and tokenizer to the access-checked immutable revision.
    snapshot = snapshot_download(settings['model_id'], revision=settings['revision'])
    provider = TransformersEpisodeProvider(settings['model_id'])
    provider._core.model_id = snapshot
    try:
        provider.load_tokenizer()
        provider.load_model()
    finally:
        provider._core.model_id = settings['model_id']
    provider.validate_ready()
    provider.warm_up()
    tasks = json.loads((ROOT / 'prompts/benchmark_v1.json').read_text())
    backend = ScopedDockerBackend(owner)
    def handle(index):
        task = tasks[index]
        session = DomainSession.from_benchmark(task['id']) if task['id'] in {'T5', 'T6'} else None
        result = run_transformers_episode(task, provider, domain_session=session,
                   sandbox_backend=backend, run_id=settings['run_id'])
        result.run_metadata['model_revision'] = settings['revision']
        result.run_metadata['tokenizer_revision'] = settings['revision']
        return result
    return handle


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement only of this coordinator's unique run checkpoint.
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(json_bytes(value) + b'\n')
    temporary.replace(path)


def run_phase2(model_id, output_root, *, preflight_fn=run_preflight,
               supervisor_factory=ProcessSupervisor, worker=official_worker,
               evaluate_fn=evaluate_episode, report_builder=build_report,
               evaluation_backend=ScopedDockerBackend, frozen_path=FROZEN, protocol_version='phase2_v1'):
    """No resume. A fresh UUID reserves exclusive artifact names for this run.

    Timeout/worker failure ends this run: the loaded CUDA process is gone. Do
    not reload mid-run or claim the remaining tasks executed. Missing episodes
    receive explicit infrastructure failures for the ten-task denominator.
    """
    preflight = preflight_fn(model_id, output_root, inspect_model=True)
    if preflight['status'] != 'PASS':
        raise RuntimeError('preflight_failed: ' + ','.join(c['code'] for c in preflight['checks']
                           if c['status'] != 'PASS'))
    tasks = json.loads((ROOT / 'prompts/benchmark_v1.json').read_text())
    if [t['id'] for t in tasks] != [f'T{i}' for i in range(1, 11)]:
        raise RuntimeError('invalid_task_order')
    run_id = uuid.uuid4().hex
    root = Path(output_root)
    raw_path = root / 'results/raw/phase2' / slug(model_id) / ('benchmark_' + run_id + '.json')
    manifest_path = root / 'reports/phase2' / slug(model_id) / (run_id + '_manifest.json')
    eval_path = manifest_path.with_name(run_id + '_evaluation.json')
    raw = {'protocol_version': protocol_version, 'phase': 'phase2', 'condition': 'tools', 'run_id': run_id, 'model_id': model_id,
           'status': 'incomplete', 'tasks': [], 'preflight': preflight,
           'configuration': {'dtype': 'float16', 'device_map': 'auto', 'max_new_tokens': 512,
              'do_sample': False, 'temperature': None, 'seed': None, 'warmups': 2,
              'max_tool_calls': 8, 'max_model_outputs': 9, 'task_timeout_seconds': 600,
              'context_tokens': 16384, 'reserved_output_tokens': 512, 'input_limit': 15872}}
    # Unique run artifacts reserved before starting expensive work.
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open('xb') as stream:
        stream.write(json_bytes(raw))
    evaluations = []
    metadata = {}
    supervisor = None
    failure = None
    diagnostic = None
    try:
        with tempfile.TemporaryDirectory(prefix='phase2-worker-') as directory:
            supervisor = supervisor_factory(worker, {'model_id': model_id, 'run_id': run_id,
                'revision': preflight['model_revision']}, directory)
            for index, task in enumerate(tasks):
                event = supervisor.execute(index, timeout=600)
                if event['supervision'] != 'completed':
                    failure = event['supervision']
                    diagnostic = event.get('failure_details')
                    break
                result = event['result']
                metadata = result.run_metadata
                raw['tasks'].append({'task_id': task['id'], **asdict(result)})
                # Evaluation uses the same secure backend, in the coordinator,
                # after the supervised episode has completed.
                evaluations.append(evaluate_fn(task, result.episode,
                                                    sandbox_backend=evaluation_backend(run_id)))
                _write(raw_path, raw)
            if len(raw['tasks']) == 10:
                raw['status'] = 'complete'
    except Exception as error:
        failure = 'infrastructure_failure'
        diagnostic = failure_details(error, stage='runner')
    finally:
        if supervisor is not None:
            try:
                supervisor.close()
            except Exception as error:
                failure = 'cleanup_failure'
                diagnostic = failure_details(error, stage='worker_cleanup')
    if failure:
        raw['status'] = 'incomplete'
        raw['failure'] = failure
    # Explicit missing/error tasks keep the complete ten-task denominator.
    first_missing = len(evaluations)
    for missing_index, task in enumerate(tasks[first_missing:]):
        reason = ('timeout' if failure == 'timeout' else 'executor_error') if missing_index == 0 and failure else 'not_executed'
        episode = EpisodeResult(termination_reason=reason, infrastructure_failure=reason != 'timeout',
                    metadata={'task_id': task['id'], 'model_id': model_id, 'run_id': run_id})
        if missing_index == 0 and diagnostic is not None:
            episode.failure_details = diagnostic
        if not any(r['task_id'] == task['id'] for r in raw['tasks']):
            raw['tasks'].append({'task_id': task['id'], 'episode': episode.to_dict(), 'turn_metrics': [],
                                 'context_checks': [], 'not_executed': reason == 'not_executed'})
        evaluations.append(evaluate_fn(task, episode))
    if protocol_version == 'phase2_v2':
        # Preserve status compatibility: complete means all ten episodes visited,
        # not that model execution succeeded or results can be published.
        raw['collection_status'] = raw['status']
        from benchmark.evaluation.phase2_v2 import collection_valid
        raw['experimental_valid'] = collection_valid(evaluations, raw['status'])
        if diagnostic is not None:
            raw['failure_details'] = diagnostic
    _write(raw_path, raw)
    environment = {'python': platform.python_version()}
    for package in ('torch', 'transformers', 'accelerate'):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    manifest = {'environment_versions': environment, 'run_id': run_id, 'status': raw['status'], 'preflight': preflight,
        'protocol_version': protocol_version, 'frozen_hashes': json.loads(Path(frozen_path).read_text()),
        'implementation_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sorted(
            list((ROOT / 'benchmark').rglob('*.py')) + [ROOT / 'run_phase2.py'])},
        'runtime': metadata, 'raw_sha256': sha(raw_path)}
    if protocol_version == 'phase2_v2':
        manifest.update(collection_status=raw['collection_status'], experimental_valid=raw['experimental_valid'])
    _write(manifest_path, manifest)
    report = report_builder(evaluations, model_id=model_id, run_metadata={'run_id': run_id},
                          source_raw_path=raw_path, **({'collection_status': raw['status'],
                          'collection_failure': failure} if protocol_version == 'phase2_v2' else {}))
    report['collection_status'] = raw['status']
    report['collection_failure'] = failure
    from benchmark.orchestration.tool_trace import finite_json_bytes
    report.pop('structural_hash')
    report['structural_hash'] = hashlib.sha256(finite_json_bytes(report)).hexdigest()
    export_report(report, eval_path)
    return {'status': raw['status'], 'raw': raw_path, 'manifest': manifest_path, 'evaluation': eval_path}
