# Operação da Fase 2

Nenhuma coleta foi realizada na preparação. O protocolo metodológico permanece
inalterado. Requer Linux, Python 3.10–3.12, CUDA e Docker **realmente funcional**.
Colab gerenciado geralmente não fornece daemon Docker/isolamento necessário:
nesse caso `safe_sandbox_unavailable` bloqueia a coleta. Use um runtime próprio
com GPU e Docker (conectável ao Colab); não substitua a sandbox por Python local.

1. Clone/pull `main`, confira o commit e use um ambiente limpo.
2. `pip install -r requirements-phase2.txt`. O arquivo de constraints prescreve
   uma toolchain explícita; ela ainda precisa passar pelo preflight no hardware
   de coleta. Versões de torch/Transformers/Accelerate da Fase 1 não foram
   registradas nos artefatos antigos: não se presume igualdade de versões.
3. Provisione Docker fora do benchmark. Baixe previamente a imagem exata indicada
   em `benchmark/tools/SANDBOX_README.md`; ferramentas nunca instalam pacotes.
4. No Colab, forneça o secret sem imprimi-lo:
   ```python
   import os
   from google.colab import userdata
   os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
   ```
5. Preflight (config/tokenizer, sem carregar pesos):
   `python run_phase2.py --mode preflight --model meta-llama/Llama-3.2-1B-Instruct`
   Qualquer FAIL impede coleta. `--mode environment` não acessa modelos e marca
   itens dependentes deles como `not_executed_until_colab`, nunca PASS.
6. Somente após revisão do PASS, execute o mesmo comando com `--mode run`.
   O preflight é repetido; a revisão do modelo é fixada por snapshot. Os outros
   IDs aceitos são os 3B/8B oficiais do protocolo; não há fallback.
7. Preserve `results/raw/phase2/<model>/benchmark_<id>.json` e os dois arquivos
   em `reports/phase2/<model>/`: manifesto e avaliação. RAW contém contexto e
   respostas experimentais, não um corpus de treino; controle sua publicação.
8. Faça revisão humana dos artefatos antes de qualquer push manual. O runner
   nunca faz Git commit/push, não sobrescreve Fase 1 e não retoma runs.

Um resultado anterior no mesmo output-root/modelo bloqueia uma coleta ambígua.
Para reiniciar um run interrompido, selecione conscientemente um novo
`--output-root` vazio fora do checkout (por exemplo, Drive); todos os dez episódios começam do zero. Timeout de 600 s
mata o grupo de processos do worker e limpa seus containers rotulados antes de
finalizar; a tarefa interrompida e as restantes ficam explicitamente censuradas.
Não há recarga automática de CUDA no mesmo run. Startup/download/carregamento e
warm-ups têm supervisão separada; não integram latência de episódio.

Smoke reproduzível, sem download/modelo, em diretório temporário:
`python -m unittest discover -s tests -p test_phase2_operations.py -v`.
