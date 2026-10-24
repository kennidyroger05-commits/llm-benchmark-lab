"""Evaluate the three frozen official RAW runs without modifying their inputs."""
import csv
import hashlib
import json
from pathlib import Path

from benchmark.evaluation.aggregation import aggregate, discover_runs
from benchmark.evaluation.parsing import parse_json
from benchmark.evaluation.task_evaluators import evaluate_task

ROOT = Path(__file__).resolve().parent


def render_report(summaries, verdicts, hardware_note):
    """Render descriptive findings only; hardware differs between model runs."""
    lines = [
        '# Fase 1 — avaliação consolidada RAW', '',
        'Protocolo: Benchmark V1 único, T1–T10 sequenciais, phase1/raw, Transformers, '
        'float16, device_map=auto, max_new_tokens=512, do_sample=False, temperature=None, '
        'seed=None e dois warm-ups descartados. Sem ferramentas ou juiz LLM.', '',
        'Acerto estrito exige sucesso técnico, formato e conteúdo corretos. O diagnóstico '
        'de conteúdo aceita somente a remoção de um fence Markdown completo; não corrige '
        'valores, tipos, campos, prosa ou código. Esse diagnóstico não conta como acerto oficial. '
        'As regras vêm do Benchmark V1 e os casos adicionais de código cobrem seu contrato.', '',
        '| Métrica | 1B | 3B | 8B |', '|---|---:|---:|---:|',
    ]
    for field, label in [('technical_success', 'Sucesso técnico'), ('strict_correct', 'Strict accuracy'),
                         ('content_correct', 'Content accuracy (diagnóstico)'),
                         ('format_correct', 'Format compliance'), ('hit_max_new_tokens', 'Hit max tokens')]:
        cells = [f'{s[field+"_count"]}/{s["total_tasks"]} ({s[field+"_rate"]:.0%})' for s in summaries.values()]
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    for field, label in [('input_tokens_total', 'Tokens de entrada'), ('output_tokens_total', 'Tokens de saída'),
                         ('total_tokens', 'Tokens totais'), ('inference_latency_total_seconds', 'Latência total (s)'),
                         ('inference_latency_mean_seconds', 'Latência média (s)'),
                         ('inference_latency_median_seconds', 'Latência mediana (s)'),
                         ('observed_throughput_total', 'Throughput observado (tokens/s)'),
                         ('model_load_time_seconds', 'Carregamento único (s)'), ('device', 'Dispositivo'),
                         ('gpu_name', 'GPU'), ('vram_gb', 'VRAM (GB)'), ('dtype', 'dtype')]:
        cells = [f'{s[field]:.4f}' if isinstance(s[field], float) else str(s[field]) for s in summaries.values()]
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'O throughput é a soma dos tokens gerados dividida pela soma das latências '
              'medidas de inferência completa; não é throughput isolado de decode. O tempo de '
              'carregamento é contado uma vez por run. Não há arredondamento nas métricas JSON.', '',
              '## Resultado por tarefa', '',
              'Cada célula registra estrito/conteúdo/formato (1=correto, 0=incorreto), seguido dos códigos de diagnóstico.', '',
              '| Tarefa | 1B | 3B | 8B |', '|---|---|---|---|']
    for index in range(10):
        cells = []
        for model in summaries:
            v = verdicts[model][index]
            flags = '/'.join(str(int(v[k])) for k in ['strict_correct', 'content_correct', 'format_correct'])
            cells.append(flags + ': ' + (', '.join(v['failure_reasons']) or 'ok'))
        lines.append(f'| T{index+1} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Limite de tokens', '',
              '`hit_max_new_tokens` marca output_tokens == max_new_tokens. É evidência de possível '
              'truncamento, sem confirmar corte semântico. Nesses casos `truncated=null`; nos demais '
              '`truncated=false` significa ausência desse indício, não prova de completude. Nenhuma '
              'resposta é completada ou reparada pelo avaliador.', '',
              '## Hardware e viabilidade', '',
              'Os runs oficiais de 1B e 3B utilizaram Tesla T4; o de 8B utilizou NVIDIA L4. '
              'Latência e throughput do 8B versus os modelos menores não representam comparação '
              'com hardware controlado. Os dados não isolam o efeito do tamanho do modelo.', '',
              'Registro da tentativa em T4 (não é um quarto run oficial):', '',
              '\n'.join('> ' + line for line in hardware_note.strip().splitlines()), '',
              '## Conclusões descritivas', '']
    for alias, s in summaries.items():
        strict_ids = [v['task_id'] for v in verdicts[alias] if v['strict_correct']]
        diagnostic_ids = [v['task_id'] for v in verdicts[alias] if v['content_correct'] and not v['strict_correct']]
        lines.append(f'- {alias.upper()}: {s["strict_correct_count"]}/10 acertos estritos; '
                     f'{s["content_correct_count"]}/10 no diagnóstico de conteúdo; '
                     f'{s["format_correct_count"]}/10 em conformidade de formato; '
                     f'{s["hit_max_new_tokens_count"]}/10 atingiram o limite de tokens. '
                     f'Tarefas com acerto estrito: {", ".join(strict_ids) or "nenhuma"}. '
                     f'Conteúdo correto sem acerto estrito: {", ".join(diagnostic_ids) or "nenhuma"}.')
    lines += ['', 'Os resultados descrevem somente estas dez tarefas e estes runs; não sustentam '
              'superioridade global nem atribuição causal. A tentativa 8B FP16 na T4 é evidência '
              'de limitação de hardware/viabilidade, excluída da acurácia.', '',
              'Não há placar manual anterior estruturado disponível no repositório para uma '
              'comparação automática de divergências. A separação entre acerto estrito e diagnóstico '
              'de conteúdo torna explícito o tratamento de Markdown e evita considerá-lo acerto oficial.', '',
              'Os casos de avaliação de código são exclusivamente de teste e devem ser excluídos '
              'de qualquer corpus de fine-tuning. Consulte benchmark/evaluation/README.md para as '
              'restrições do processo de execução.', '']
    return '\n'.join(lines)


def main(root=ROOT):
    root = Path(root)
    benchmark_path = root / 'prompts/benchmark_v1.json'
    tasks = parse_json(benchmark_path.read_text(encoding='utf-8'))
    runs = discover_runs(root / 'results/raw/phase1', tasks)
    note_path = root / 'reports/phase1/hardware/8b_t4_fp16_attempt.md'
    note = note_path.read_text(encoding='utf-8')
    summaries, verdicts, sources = {}, {}, {}
    for alias, (path, rows) in runs.items():
        verdicts[alias] = [evaluate_task(task, row) for task, row in zip(tasks, rows)]
        summaries[alias] = aggregate(rows, verdicts[alias])
        sources[alias] = {'path': path.relative_to(root).as_posix(),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    artifact = {
        'evaluation_version': 1, 'phase': 'phase1', 'condition': 'raw',
        'benchmark_sha256': hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        'evaluator_sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted([*(root / 'benchmark/evaluation').glob('*.py'),
                                              *root.glob('evaluate_phase1.py')])},
        'sources': sources, 'summaries': summaries, 'task_results': verdicts,
        'truncated_policy': 'null at token cap (suspected only); false otherwise (no token-cap evidence)',
        'hardware_attempt_sha256': hashlib.sha256(note_path.read_bytes()).hexdigest(),
    }
    destination = root / 'results/processed/phase1'
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / 'phase1_evaluation.json'
    csv_path = destination / 'phase1_task_results.csv'
    report_path = root / 'reports/phase1/phase1_summary.md'
    # These consolidated outputs intentionally regenerate the same paths and bytes.
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    flattened = [v for group in verdicts.values() for v in group]
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        for row in flattened:
            writer.writerow({**row, 'failure_reasons': json.dumps(row['failure_reasons'])})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summaries, verdicts, note), encoding='utf-8')
    for alias, summary in summaries.items():
        print(f'{alias}: strict={summary["strict_correct_rate"]:.0%}, '
              f'content={summary["content_correct_rate"]:.0%}, '
              f'format={summary["format_correct_rate"]:.0%}, '
              f'hit_max={summary["hit_max_new_tokens_count"]}')
    for path in [json_path, csv_path, report_path]:
        print(path.relative_to(root))
    return artifact


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f'Avaliação interrompida: {error}') from None
