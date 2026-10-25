# Phase 2 V2 — tentativas inválidas de infraestrutura e memória

Checkpoint investigado: `afda967`. Nenhum modelo oficial foi executado nesta
investigação; testes de geração usam mocks e uma fixture CPU local minúscula com
pesos inicializados localmente, sem download de modelo/tokenizer oficial.

## Preservação das tentativas

- `b272308bed4c438cbb38176bdf33a512`: dez `model_failure`, zero tokens gerados;
  exceção original indisponível no registro anterior ao hotfix. O antigo
  `publication_ready=true` era um bug; a tentativa é inválida e não é score zero.
- `59caa30bc8ef4080801713cf2e96b234`: usuário confirmou `OutOfMemoryError`,
  `model_generate`, turno 1 em T1–T10, zero tokens gerados, dez infrastructure
  failures; `experimental_valid=false` e `publication_ready=false`.

O segundo run usou Llama 3.2 1B Instruct, FP16, Tesla T4 (~14,563 GiB), cuda:0,
sem offload; havia ~2,7–2,9 GiB livres. Warm-ups concluíram. O smoke curto passou
(78 tokens de entrada, 12 de saída, ~0,66 s). Os RAWs/manifests originais não foram
anexados nem regravados: estas observações foram transcritas pelo usuário.
Ambas as tentativas ficam excluídas de qualquer conclusão de acurácia.

## Contexto da primeira chamada

Tokens abaixo foram fornecidos pelo usuário a partir do RAW. Capacidade declarada
131072, política experimental 16384, reserva 512, entrada máxima 15872. Total
86310 tokens de entrada, zero de saída. O orçamento oficial continua 512 por turno.
Os tamanhos UTF-8 foram calculados offline com o builder e catálogo do checkpoint;
incluem a serialização JSON de cada valor, mas não o chat template. Chaves,
separadores e template explicam o restante. Não foram estimados tokens por
caracteres nem atribuídos tokens aos componentes: BPE não é aditivo nas fronteiras.

| ID | Tokens reais de entrada | Protocolo bytes | Catálogo bytes | Tarefa bytes | Histórico bytes | Documento bytes | Softmax FP32 GiB (teórico) | KV até N+512 GiB (teórico) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 8493 | 639 | 33405 | 745 | 2 | 34839 | 8.599 | 0.275 |
| T2 | 8448 | 639 | 33405 | 639 | 2 | 34733 | 8.508 | 0.273 |
| T3 | 8616 | 639 | 33405 | 936 | 2 | 35030 | 8.850 | 0.279 |
| T4 | 8793 | 639 | 33405 | 1737 | 2 | 35831 | 9.217 | 0.284 |
| T5 | 8742 | 639 | 33405 | 1870 | 2 | 35964 | 9.110 | 0.282 |
| T6 | 8813 | 639 | 33405 | 2079 | 2 | 36173 | 9.259 | 0.285 |
| T7 | 8515 | 639 | 33405 | 957 | 2 | 35051 | 8.643 | 0.275 |
| T8 | 8548 | 639 | 33405 | 1024 | 2 | 35118 | 8.710 | 0.276 |
| T9 | 8692 | 639 | 33405 | 1427 | 2 | 35521 | 9.006 | 0.281 |
| T10 | 8650 | 639 | 33405 | 1365 | 2 | 35459 | 8.920 | 0.280 |

Catálogo serializado: 33405 bytes, inalterado. Hash V2:
`f802970524fa3daadb8e30801aa40ed81995b1b8d0f1fd64053e34b178ddc112`.

## Explicação da alocação

Para batch 1, 32 cabeças de atenção, N tokens, a matriz de softmax FP32 ocupa
`1 * 32 * N * N * 4` bytes. T1 dá 8,599 GiB; T6 dá 9,259 GiB: coincide com os
OOMs relatados. Scores FP16 ocupam metade disso e podem coexistir. O KV cache
é linear em N: `2 * 16 layers * 8 KV heads * (N+512) * 64 * 2` bytes, cerca de
0,28 GiB. Esses números assumem as dimensões do 1B; não representam o pico
completo do processo nem explicam toda a VRAM já ocupada antes do prefill.

Fontes inspecionadas, sem acessar modelo oficial:

- Transformers 4.57.6, `integrations/sdpa_attention.py`: com máscara None e
  torch>=2.5, `use_gqa_in_sdpa` ativa `enable_gqa=True` em Llama/GQA.
- PyTorch v2.11.0, `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`:
  kernel eficiente rejeita números diferentes de cabeças Q e K/V; Flash e
  cuDNN exigem SM80+, enquanto T4 é SM75. Resta o caminho matemático quadrático.
- Transformers 4.57.6, `models/llama/modeling_llama.py`: eager também materializa
  scores e softmax FP32 quadráticos. O tamanho da alocação sozinho não distingue
  eager de SDPA/math. SDPA/math é o caminho inferido dos defaults e do hardware;
  `attention_impl` anterior não foi registrado, portanto não é observação direta.
- `generation/utils.py`: geração usa `logits_to_keep=1` quando suportado e
  DynamicCache por default; cache real anterior também não foi registrado.

Não há código local forçando eager, máscara 4D, attentions/scores ou StaticCache.
A atenção causal/padding e cache_position vêm do Transformers. O código roda
sob `torch.no_grad()`. O catálogo/histórico ficam em strings CPU; inputs preparados
ficam em CPU e são movidos para o dispositivo de execução. Não há retenção de
logits, KV ou histórico GPU em EpisodeResult. O novo cleanup libera também o cache
CPU de tokenização após erros; `empty_cache()` não é tratado como solução do O(N²).

## Correção de execução

Apenas Phase 2 registra o alias HF `phase2_sdpa_expanded_kv` e reutiliza o builder
original de máscaras SDPA. K/V são repetidos por cabeça explicitamente, como na
atenção eager/GQA de referência; chama-se SDPA com `enable_gqa=False`. Isso evita
que a T4 seja forçada ao backend matemático pelo GQA nativo. Em CUDA permitem-se
somente os kernels efficient/Flash: ausência de kernel compatível é falha clara,
sem retry, math fallback, CPU fallback, offload novo ou mudança de dtype.
O nome padrão `sdpa` global não é sobrescrito; Phase 1 permanece intacta.

Cache e kwargs oficiais não são alterados. Novos RAWs registram configuração
allowlisted, classe efetiva do cache no primeiro forward, shapes/dtypes de input,
attention_mask e cache_position, flags, memória CUDA antes/depois e picos. O hook
não guarda tensors e é removido mesmo em erro. O cache automático continua sendo
registrado como null/default até a classe ser realmente observada.

Validação CPU com torch 2.11.0+cpu e Transformers 4.57.6: equivalência numérica do
SDPA com expansão explícita; fixture local de uma camada com logits equivalentes
e DynamicCache. CUDA/T4 não está disponível aqui; o kernel eficiente e a redução
de pico ainda precisam de confirmação por smoke real autorizado.

## Smoke longo não experimental

```bash
python run_phase2.py --mode provider-context-smoke --model meta-llama/Llama-3.2-1B-Instruct --output-root /tmp/phase2-context-smoke
```

O preflight V2 normal continua bloqueante e lê bytes congelados só para hashes.
O smoke não desserializa tarefas nem usa conteúdo do benchmark. Monta texto
sintético até ~8700 tokens, usando contagem real após template (tolerância 16),
sem truncar. Opcional: `--context-smoke-input-tokens 8813` para aproximar T6.
Faz no máximo uma geração de 8 tokens pelo mesmo provider; nenhum warm-up,
loop de tarefas, tools, avaliação ou arquivo oficial. O orçamento diagnóstico
não altera os 512 tokens por turno da coleta. Entrada máxima continua 15872.

JSON stdout inclui PASS/FAIL, tokens, latência, picos allocated/reserved e VRAM
livre antes/depois (bytes, dispositivo explícito), alias de atenção, cache,
observações do primeiro forward e failure_details sanitizado. Sem HF_TOKEN,
headers ou env. Use diretório novo se houver conflito de run no output-root.

A correção não reduz catálogo, prompts, contexto, modelos ou limites 8 tools/9
outputs e não muda métricas funcionais. As tentativas anteriores não são
reclassificadas como resultados válidos por esta alteração.
