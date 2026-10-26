# Fase 2 — avaliação consolidada com ferramentas

Protocolo: Benchmark V1 único, T1–T10 sequenciais, `phase2/tools`, Transformers, `float16`, `device_map=auto`, `max_new_tokens=512`, `do_sample=False`, `temperature=None`, `seed=None` e dois warm-ups descartados. O catálogo global contém 25 ferramentas e é idêntico para todos os modelos e tarefas. O protocolo V2 exige que cada saída do modelo seja **somente JSON**, em um dos envelopes `tool_call` ou `final_answer`, com no máximo 8 chamadas despachadas e 9 saídas de modelo por tarefa. O sandbox é `restricted_ast_interpreter`. Não há juiz LLM.

A Fase 2 mede se a simples disponibilização das ferramentas, acompanhada apenas do prompt operacional mínimo necessário para expor catálogo e protocolo, é suficiente para produzir uso funcional de ferramentas. Não há engenharia deliberada de prompt nesta fase.

A métrica funcional primária é `task_correct`. `strict_correct`, `content_correct`, `format_correct` e `protocol_correct` são mantidas separadamente para diagnóstico. `technical_success` mede integridade de execução/infraestrutura e não deve ser confundido com acerto da tarefa. Uma resposta pode ser tecnicamente gerada e ainda falhar no protocolo.

Runs oficiais analisados:

- 1B: `566b644cfa2b48ad8817cb66719817ba`
- 3B: `dacf4f80de984a7191662b2d92f4d2fe`
- 8B: `df48aa699e9e4cfd9317d99514c6d7b1`

Todos os três runs têm `collection_status=complete`, `experimental_valid=true` e `publication_ready=true`.

## Métricas consolidadas

| Métrica | 1B | 3B | 8B |
|---|---:|---:|---:|
| Sucesso técnico | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Task accuracy (primária) | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |
| Strict accuracy | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |
| Content accuracy | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |
| Format compliance | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |
| Protocol compliance | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |
| Tool calls despachadas | 0 | 0 | 0 |
| Tool calls totais registradas | 0 | 0 | 0 |
| Seleção `invalid` | 2/10 | 2/10 | 2/10 |
| Seleção `unclassified` | 8/10 | 8/10 | 8/10 |
| Hit max tokens | 8/10 (80%) | 4/10 (40%) | 9/10 (90%) |
| Tokens de entrada | 86310 | 86310 | 86310 |
| Tokens de saída | 4810 | 3535 | 5033 |
| Tokens totais | 91120 | 89845 | 91343 |
| Latência total de geração (s) | 190.3199 | 375.1634 | 612.7404 |
| Latência média por tarefa (s) | 19.0320 | 37.5163 | 61.2740 |
| Latência mediana por tarefa (s) | 20.1785 | 40.9223 | 62.1659 |
| Throughput observado (tokens/s) | 25.2732 | 9.4226 | 8.2139 |
| Carregamento único (s) | 10.1134 | 31.9489 | 9.7668 |
| Dispositivo | cuda:0 | cuda:0 | cuda:0 |
| GPU | Tesla T4 | Tesla T4 | NVIDIA L4 |
| VRAM (GB) | 14.5632 | 14.5632 | 22.0343 |
| dtype | float16 | float16 | float16 |

O throughput é a soma dos tokens gerados dividida pela soma das latências medidas de inferência completa; não é throughput isolado de decode. O tempo de carregamento é contado uma vez por run. Os valores originais permanecem sem arredondamento nos JSONs de manifesto/RAW; a tabela acima arredonda somente para leitura.

## Resultado por tarefa

Cada célula registra `task/strict/protocolo/conteúdo/formato` (1=correto, 0=incorreto), seguido do motivo de término. Nenhuma ferramenta foi despachada em qualquer tarefa.

| Tarefa | 1B | 3B | 8B |
|---|---|---|---|
| T1 | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls |
| T2 | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls |
| T3 | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls |
| T4 | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls |
| T5 | 0/0/0/0/0: `invalid_protocol`, selection=`invalid`, 0 calls | 0/0/0/0/0: `invalid_protocol`, selection=`invalid`, 0 calls | 0/0/0/0/0: `invalid_protocol`, selection=`invalid`, 0 calls |
| T6 | 0/0/0/0/0: `invalid_protocol`, selection=`invalid`, 0 calls | 0/0/0/0/0: `invalid_protocol`, selection=`invalid`, 0 calls | 0/0/0/0/0: `invalid_protocol`, selection=`invalid`, 0 calls |
| T7 | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls |
| T8 | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls |
| T9 | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls |
| T10 | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls | 0/0/0/0/0: `invalid_protocol`, 0 calls |

Nos três modelos, T5 e T6 são as duas tarefas multi-tool/domain e recebem `tool_selection_class=invalid`; as demais oito ficam `unclassified`, porque o episódio termina antes de uma chamada válida de ferramenta ser despachada.

## Limite de tokens

`hit_max_new_tokens` marca `output_tokens == max_new_tokens` na geração observada e é evidência de possível truncamento, sem provar corte semântico.

- 1B: T1, T2, T4, T5, T7, T8, T9 e T10 atingiram 512 tokens (8/10).
- 3B: T2, T3, T5 e T6 atingiram 512 tokens (4/10).
- 8B: T1, T2, T3, T4, T6, T7, T8, T9 e T10 atingiram 512 tokens (9/10).

O fato de uma resposta atingir o limite não altera a causa registrada pelo avaliador: todos os episódios terminaram por `invalid_protocol` já na primeira saída do modelo.

## Uso de ferramentas e falha de protocolo

O resultado central da Fase 2 é que **nenhum dos três modelos chegou a executar uma ferramenta**. Todos produziram uma primeira saída de modelo com sucesso técnico, porém incompatível com o envelope JSON estrito exigido pelo protocolo V2. Consequentemente:

- `protocol_correct=0/10` para 1B, 3B e 8B;
- `tool_calls_dispatched=0` para todos;
- o catálogo e o sandbox não chegaram a ser exercitados funcionalmente pelos modelos nestes runs;
- não há base nestes runs para comparar qualidade de execução de ferramentas entre 1B, 3B e 8B, porque a falha ocorreu antes do despacho.

Os RAWs mostram comportamento descritivo compatível com esse diagnóstico: em vez de emitir exclusivamente um único objeto JSON de protocolo, os modelos frequentemente produziram explicações, Markdown, código ou exemplos de chamadas. Essa observação é descritiva; o resultado oficial continua sendo o código determinístico `invalid_protocol` do avaliador.

## Hardware e viabilidade

Os runs oficiais de 1B e 3B utilizaram Tesla T4; o de 8B utilizou NVIDIA L4. Portanto, latência e throughput do 8B versus os modelos menores não representam uma comparação com hardware controlado e não isolam o efeito do tamanho do modelo.

Todos os runs oficiais desta avaliação usaram `float16`, sem offload observado, `DynamicCache` via política padrão do Transformers e a implementação de atenção `phase2_sdpa_expanded_kv`. O contexto total permitido foi 16384 tokens, com 512 reservados para saída e limite de entrada de 15872 tokens. Os 10 prompts oficiais somaram 86310 tokens de entrada em cada modelo, devido principalmente ao catálogo global de ferramentas incluído em cada tarefa.

## Conclusões descritivas

- 1B: 10/10 sucessos técnicos; 0/10 `task_correct`; 0/10 protocolo; 0 ferramentas despachadas; 8/10 gerações atingiram o limite de 512 tokens.
- 3B: 10/10 sucessos técnicos; 0/10 `task_correct`; 0/10 protocolo; 0 ferramentas despachadas; 4/10 gerações atingiram o limite de 512 tokens.
- 8B: 10/10 sucessos técnicos; 0/10 `task_correct`; 0/10 protocolo; 0 ferramentas despachadas; 9/10 gerações atingiram o limite de 512 tokens.

Dentro destas dez tarefas e destes runs, aumentar o tamanho do modelo de 1B para 3B ou 8B **não foi suficiente para superar a barreira de aderência ao protocolo sob o prompt operacional mínimo da Fase 2**. Esse resultado não demonstra incapacidade geral dos modelos para tool use; demonstra somente que, nesta configuração congelada, disponibilizar catálogo e protocolo sem engenharia deliberada de prompting/orquestração não produziu uso funcional das ferramentas.

A Fase 3 deve testar explicitamente se prompting/orquestração melhora a aderência ao protocolo e o uso das ferramentas. Para preservar validade experimental, variantes de prompt não devem ser ajustadas contra T1–T10 ou seus gabaritos oficiais; o desenvolvimento deve usar tarefas separadas/held-out ou variantes pré-registradas e congeladas antes dos runs oficiais.

## Fontes dos dados

- `results/raw/phase2/meta-llama--Llama-3.2-1B-Instruct/benchmark_566b644cfa2b48ad8817cb66719817ba.json`
- `reports/phase2/meta-llama--Llama-3.2-1B-Instruct/566b644cfa2b48ad8817cb66719817ba_manifest.json`
- `reports/phase2/meta-llama--Llama-3.2-1B-Instruct/566b644cfa2b48ad8817cb66719817ba_evaluation.json`
- `results/raw/phase2/meta-llama--Llama-3.2-3B-Instruct/benchmark_dacf4f80de984a7191662b2d92f4d2fe.json`
- `reports/phase2/meta-llama--Llama-3.2-3B-Instruct/dacf4f80de984a7191662b2d92f4d2fe_manifest.json`
- `reports/phase2/meta-llama--Llama-3.2-3B-Instruct/dacf4f80de984a7191662b2d92f4d2fe_evaluation.json`
- `results/raw/phase2/meta-llama--Llama-3.1-8B-Instruct/benchmark_df48aa699e9e4cfd9317d99514c6d7b1.json`
- `reports/phase2/meta-llama--Llama-3.1-8B-Instruct/df48aa699e9e4cfd9317d99514c6d7b1_manifest.json`
- `reports/phase2/meta-llama--Llama-3.1-8B-Instruct/df48aa699e9e4cfd9317d99514c6d7b1_evaluation.json`

Os resultados descrevem somente estas dez tarefas e estes runs; não sustentam superioridade global nem atribuição causal sobre capacidade geral de modelos.