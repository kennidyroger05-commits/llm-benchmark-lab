# Fase 1 — avaliação consolidada RAW

Protocolo: Benchmark V1 único, T1–T10 sequenciais, phase1/raw, Transformers, float16, device_map=auto, max_new_tokens=512, do_sample=False, temperature=None, seed=None e dois warm-ups descartados. Sem ferramentas ou juiz LLM.

Acerto estrito exige sucesso técnico, formato e conteúdo corretos. O diagnóstico de conteúdo aceita somente a remoção de um fence Markdown completo; não corrige valores, tipos, campos, prosa ou código. Esse diagnóstico não conta como acerto oficial. As regras vêm do Benchmark V1 e os casos adicionais de código cobrem seu contrato.

| Métrica | 1B | 3B | 8B |
|---|---:|---:|---:|
| Sucesso técnico | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Strict accuracy | 0/10 (0%) | 1/10 (10%) | 1/10 (10%) |
| Content accuracy (diagnóstico) | 0/10 (0%) | 1/10 (10%) | 2/10 (20%) |
| Format compliance | 0/10 (0%) | 2/10 (20%) | 2/10 (20%) |
| Hit max tokens | 4/10 (40%) | 0/10 (0%) | 2/10 (20%) |
| Tokens de entrada | 4227 | 4227 | 4227 |
| Tokens de saída | 3312 | 1289 | 2684 |
| Tokens totais | 7539 | 5516 | 6911 |
| Latência total (s) | 81.8066 | 55.9005 | 172.9258 |
| Latência média (s) | 8.1807 | 5.5901 | 17.2926 |
| Latência mediana (s) | 9.2655 | 6.2930 | 14.7084 |
| Throughput observado (tokens/s) | 40.4857 | 23.0588 | 15.5211 |
| Carregamento único (s) | 5.2595 | 91.8900 | 56.6234 |
| Dispositivo | cuda:0 | cuda:0 | cuda:0 |
| GPU | Tesla T4 | Tesla T4 | NVIDIA L4 |
| VRAM (GB) | 14.5632 | 14.5632 | 22.0343 |
| dtype | float16 | float16 | float16 |

O throughput é a soma dos tokens gerados dividida pela soma das latências medidas de inferência completa; não é throughput isolado de decode. O tempo de carregamento é contado uma vez por run. Não há arredondamento nas métricas JSON.

## Resultado por tarefa

Cada célula registra estrito/conteúdo/formato (1=correto, 0=incorreto), seguido dos códigos de diagnóstico.

| Tarefa | 1B | 3B | 8B |
|---|---|---|---|
| T1 | 0/0/0: invalid_json, markdown_not_allowed, wrong_type | 0/0/0: invalid_json, markdown_not_allowed, wrong_value | 0/0/0: invalid_json, markdown_not_allowed, wrong_type |
| T2 | 0/0/0: invalid_json, markdown_not_allowed, wrong_type | 0/0/0: invalid_json, markdown_not_allowed, wrong_value | 0/0/0: hit_max_new_tokens, invalid_json, markdown_not_allowed, wrong_type |
| T3 | 0/0/0: hit_max_new_tokens, invalid_json, markdown_not_allowed, wrong_type | 0/0/0: invalid_json, markdown_not_allowed, wrong_value | 0/0/0: invalid_json, markdown_not_allowed, wrong_type |
| T4 | 0/0/0: hit_max_new_tokens, invalid_json, markdown_not_allowed, wrong_type | 0/0/0: invalid_json, markdown_not_allowed, wrong_type | 0/0/0: hit_max_new_tokens, invalid_json, markdown_not_allowed, wrong_type |
| T5 | 0/0/0: extra_fields, extra_tool_call, wrong_tool_arguments | 0/0/0: invalid_json, wrong_type | 0/0/1: extra_tool_call |
| T6 | 0/0/0: extra_fields | 1/1/1: ok | 1/1/1: ok |
| T7 | 0/0/0: markdown_not_allowed, wrong_code_behavior | 0/0/0: markdown_not_allowed, wrong_code_behavior | 0/1/0: markdown_not_allowed |
| T8 | 0/0/0: hit_max_new_tokens, invalid_python, markdown_not_allowed | 0/0/0: markdown_not_allowed, wrong_code_behavior | 0/0/0: markdown_not_allowed, wrong_code_behavior |
| T9 | 0/0/0: hit_max_new_tokens, invalid_json, markdown_not_allowed, wrong_type | 0/0/0: invalid_json, markdown_not_allowed, wrong_value | 0/0/0: invalid_json, markdown_not_allowed, wrong_type, wrong_value |
| T10 | 0/0/0: wrong_line_format, wrong_value | 0/0/1: wrong_value | 0/0/0: wrong_line_format, wrong_value |

## Limite de tokens

`hit_max_new_tokens` marca output_tokens == max_new_tokens. É evidência de possível truncamento, sem confirmar corte semântico. Nesses casos `truncated=null`; nos demais `truncated=false` significa ausência desse indício, não prova de completude. Nenhuma resposta é completada ou reparada pelo avaliador.

## Hardware e viabilidade

Os runs oficiais de 1B e 3B utilizaram Tesla T4; o de 8B utilizou NVIDIA L4. Latência e throughput do 8B versus os modelos menores não representam comparação com hardware controlado. Os dados não isolam o efeito do tamanho do modelo.

Registro da tentativa em T4 (não é um quarto run oficial):

> # Tentativa do Llama 3.1 8B FP16 na Tesla T4
> 
> ## Configuração
> 
> - **Modelo:** `meta-llama/Llama-3.1-8B-Instruct`
> - **Fase:** `phase1`
> - **Condição:** `raw`
> - **dtype:** `float16`
> - **device_map:** `auto`
> - **GPU:** Tesla T4
> - **VRAM disponível observada:** aproximadamente 14.56 GB
> 
> ## Observação
> 
> Os pesos do modelo foram baixados e carregados. Durante o carregamento, Transformers/Accelerate informou que parâmetros foram offloaded para a CPU. A inferência tornou-se impraticavelmente lenta; após mais de 16 minutos, o benchmark ainda não havia concluído. A execução foi interrompida manualmente e terminou com `KeyboardInterrupt`.
> 
> ## Resultado e interpretação
> 
> Nenhum arquivo JSON ou CSV oficial foi produzido. Esta tentativa não deve ser contabilizada como resultado de acurácia. Deve ser considerada evidência de limitação de hardware e de viabilidade do modelo 8B em FP16 na Tesla T4.

## Conclusões descritivas

- 1B: 0/10 acertos estritos; 0/10 no diagnóstico de conteúdo; 0/10 em conformidade de formato; 4/10 atingiram o limite de tokens. Tarefas com acerto estrito: nenhuma. Conteúdo correto sem acerto estrito: nenhuma.
- 3B: 1/10 acertos estritos; 1/10 no diagnóstico de conteúdo; 2/10 em conformidade de formato; 0/10 atingiram o limite de tokens. Tarefas com acerto estrito: T6. Conteúdo correto sem acerto estrito: nenhuma.
- 8B: 1/10 acertos estritos; 2/10 no diagnóstico de conteúdo; 2/10 em conformidade de formato; 2/10 atingiram o limite de tokens. Tarefas com acerto estrito: T6. Conteúdo correto sem acerto estrito: T7.

Os resultados descrevem somente estas dez tarefas e estes runs; não sustentam superioridade global nem atribuição causal. A tentativa 8B FP16 na T4 é evidência de limitação de hardware/viabilidade, excluída da acurácia.

Não há placar manual anterior estruturado disponível no repositório para uma comparação automática de divergências. A separação entre acerto estrito e diagnóstico de conteúdo torna explícito o tratamento de Markdown e evita considerá-lo acerto oficial.

Os casos de avaliação de código são exclusivamente de teste e devem ser excluídos de qualquer corpus de fine-tuning. Consulte benchmark/evaluation/README.md para as restrições do processo de execução.
