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
