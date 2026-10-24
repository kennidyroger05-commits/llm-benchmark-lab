"""Discover unambiguous official runs and aggregate measurements once per run."""
import json
from pathlib import Path
from statistics import mean, median

MODELS = {
    '1b': 'meta-llama/Llama-3.2-1B-Instruct',
    '3b': 'meta-llama/Llama-3.2-3B-Instruct',
    '8b': 'meta-llama/Llama-3.1-8B-Instruct',
}


def discover_runs(root, tasks):
    """Read JSON only, refusing missing, duplicate or inconsistent official runs."""
    candidates = {model: [] for model in MODELS.values()}
    ids = [task['id'] for task in tasks]
    if ids != [f'T{i}' for i in range(1, 11)]:
        raise ValueError('Benchmark deve conter T1–T10, nessa ordem.')
    for path in sorted(Path(root).rglob('*.json')):
        rows = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
            raise ValueError(f'Run JSON inválido: {path}')
        model = rows[0].get('model_id')
        if model not in candidates:
            raise ValueError(f'Modelo oficial desconhecido em {path}: {model}')
        if any(r.get('model_id') != model for r in rows):
            raise ValueError(f'model_id inconsistente: {path}')
        if [r.get('task_id') for r in rows] != ids:
            raise ValueError(f'Run deve conter exatamente T1–T10 em ordem: {path}')
        for task, row in zip(tasks, rows):
            if row.get('phase') != 'phase1' or row.get('condition') != 'raw':
                raise ValueError(f'Condição incompatível: {path}')
            prompt = task['raw_prompt'] if task['id'] in {'T5', 'T6'} else task['prompt']
            if row.get('prompt') != prompt:
                raise ValueError(f'Prompt difere do Benchmark V1: {path}, {task["id"]}')
        candidates[model].append((path, rows))
    for model, runs in candidates.items():
        if len(runs) != 1:
            raise ValueError(f'{model}: esperado exatamente 1 run JSON oficial; encontrados {len(runs)}.')
    return {alias: candidates[model][0] for alias, model in MODELS.items()}


def aggregate(rows, verdicts):
    """No rounding in JSON; repeated loading/hardware metadata must agree."""
    summary = {'model_id': rows[0]['model_id'], 'total_tasks': len(rows)}
    for field in ['technical_success', 'strict_correct', 'content_correct',
                  'format_correct', 'hit_max_new_tokens']:
        count = sum(v[field] for v in verdicts)
        summary[field + '_count'] = count
        summary[field + '_rate'] = count / len(rows)
    for source, target in [('input_tokens', 'input_tokens_total'),
                           ('output_tokens', 'output_tokens_total'), ('total_tokens', 'total_tokens')]:
        values = [r.get(source) for r in rows]
        summary[target] = sum(values) if all(v is not None for v in values) else None
    latencies = [r.get('inference_latency_seconds') for r in rows]
    complete = all(v is not None and v >= 0 for v in latencies)
    summary['inference_latency_total_seconds'] = sum(latencies) if complete else None
    summary['inference_latency_mean_seconds'] = mean(latencies) if complete else None
    summary['inference_latency_median_seconds'] = median(latencies) if complete else None
    total = summary['inference_latency_total_seconds']
    summary['observed_throughput_total'] = (summary['output_tokens_total'] / total
                                          if total and summary['output_tokens_total'] is not None else None)
    for field in ['model_load_time_seconds', 'device', 'gpu_name', 'vram_gb', 'dtype']:
        values = [r.get(field) for r in rows]
        if any(v != values[0] for v in values):
            raise ValueError(f'Metadado inconsistente no run: {field}')
        summary[field] = values[0]
    return summary
