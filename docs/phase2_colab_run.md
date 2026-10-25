# Operação da Fase 2 — Protocol V2

V1 e seu procedimento original estão preservados em `phase2_protocol.md` e
`phase2_colab_run_v1.md`. Consulte `phase2_protocol_v2.md` antes da primeira coleta.
Nenhum resultado oficial foi coletado nesta preparação.

1. Clone/pull o commit V2 aprovado. Python 3.10–3.13, Linux e GPU CUDA.
2. Instale `pip install -r requirements-phase2-v2.txt`. A faixa de compatibilidade
   não presume versões iguais à Fase 1: registre as versões reais no manifesto.
3. Forneça HF_TOKEN pelo secret/login HF existente, sem imprimi-lo. No Colab:
   ```python
   import os
   from google.colab import userdata
   os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
   ```
4. `python run_phase2.py --mode environment --model meta-llama/Llama-3.2-1B-Instruct`
   não acessa Llama; checks de modelo ficam not_executed_until_colab (FAIL global).
5. `python run_phase2.py --mode preflight --model meta-llama/Llama-3.2-1B-Instruct`.
   Exige acesso/config/tokenizer/contexto, CUDA/FP16, integridade e self-tests AST.
   Docker não é instalado, usado ou exigido. Não há execução arbitrária Python
   no host: o código dos tools é interpretado por handlers restritos.
6. Somente com todos os checks PASS e autorização de coleta, troque para `--mode run`.
   Modelos 3B/8B oficiais usam o mesmo protocolo/configuração. Não há fallback,
   quantização, retries de conteúdo ou retomada automática.
7. Preserve RAW, manifesto e avaliação em `results/raw/phase2` e `reports/phase2`.
   Revise `evaluation_pending`/`publication_ready`: respostas de código fora do
   subset precisam de correção isolada posterior, não recebem erro automático.
8. Timeout externo termina o worker e encerra o run como incompleto. Para outra
   tentativa, novo run/output-root vazio fora do checkout, sempre T1–T10 do início.
   Não misture tentativas. Publicação/Git push só após revisão humana separada.

Smoke sem modelo nem Docker:
`python -m unittest discover -s tests -p test_restricted_python_v2.py -v`.
A suíte completa preserva também testes históricos Docker V1; em Colab sem Docker
eles são marcados explicitamente como V1/skip, não contados como PASS V2.

## Smoke real do provider (não experimental)

Após instalar as dependências e configurar a autenticação, execute explicitamente:

```bash
python run_phase2.py --mode provider-smoke --model meta-llama/Llama-3.2-1B-Instruct
```

Esse modo exige o preflight V2 normal com todos os checks PASS antes de carregar
pesos, usando a mesma revisão, provider, tokenização e geração da coleta. Faz
**uma única geração sintética**, sem warm-ups/retries, ferramentas ou episódios.
Não carrega tarefas/prompts do Benchmark V1; o preflight lê seus bytes apenas
para verificar integridade. Não grava RAW, avaliação ou resultado experimental.
O comando imprime um JSON diagnóstico, identifica `experimental_result=false`
e retorna código 0 (PASS) ou 1 (FAIL); erros e preview são sanitizados.
Texto não vazio basta para PASS, sem avaliar semanticamente `SMOKE_OK`.

Se já houver um run no diretório, mantenha-o e forneça `--output-root` apontando
para um diretório novo/vazio: a checagem normal de conflitos continua bloqueante.
Esse diretório serve somente às checagens de escrita/conflito do preflight.
O smoke não substitui a autorização de coleta nem altera os dois warm-ups
obrigatórios do run oficial. Não execute esse modo durante verificações sem modelo.

### Smoke sintético de contexto grande

```bash
python run_phase2.py --mode provider-context-smoke --model meta-llama/Llama-3.2-1B-Instruct --output-root /tmp/phase2-context-smoke
```

Após o mesmo preflight/carregamento, esse diagnóstico usa contagem real do
tokenizer para construir ~8700 tokens sintéticos e gera no máximo 8 tokens,
sem acessar tarefas ou produzir resultados oficiais. O alvo pode ser ajustado
com `--context-smoke-input-tokens`, até 15872; não há truncagem nem estimativa
por caracteres. O JSON registra memória CUDA em bytes, latência, configuração
de atenção/cache e detalhes sanitizados de falha. O limite oficial continua 512
tokens de geração, com os mesmos prompts, catálogo e contexto de 16384.

Phase 2 usa o alias `phase2_sdpa_expanded_kv`: SDPA com expansão explícita K/V,
sem GQA nativo e sem fallback matemático em CUDA. A máscara original SDPA é
preservada; nenhum parâmetro de cache, dtype ou placement é trocado. Se efficient/
Flash não operar no ambiente, o diagnóstico falha sem contornar o bloqueio.
Consulte `reports/phase2/hardware/context_oom_attempts.md` para as tentativas
inválidas anteriores, evidências e limites desta investigação. Execute o smoke
com modelo real somente em etapa autorizada de diagnóstico/coleta.
