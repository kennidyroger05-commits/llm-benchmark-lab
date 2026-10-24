# Avaliação determinística da Fase 1

Execute `python evaluate_phase1.py` na raiz. A descoberta exige um JSON por model_id oficial, dez IDs ordenados e prompts RAW idênticos ao Benchmark V1. Não seleciona automaticamente entre runs duplicados. RAW e benchmark são lidos apenas; os três artefatos consolidados são regenerados nos caminhos fixos solicitados. Não há acesso a modelos ou APIs.

O JSON completo é validado sem coerção, chaves duplicadas ou números não finitos. Conteúdo é comparado com tipos, campos e ordem; somente um fence completo pode ser removido para diagnóstico. Formato é avaliado separadamente do valor. `strict_correct` exige também sucesso técnico. `truncated=null` indica que o limite de tokens foi atingido e o corte semântico é indeterminado; `false` indica somente ausência desse indício. Não há recuperação de JSON incompleto.

T7/T8 usam os casos do dataset e casos adicionais do domínio, verificando tipos, valores e não mutação da entrada. **Estes casos são exclusivos da avaliação e nunca devem integrar dados de fine-tuning.** O código de referência permanece no dataset como fonte única; não é duplicado no avaliador.

A execução usa processo Python isolado, ambiente sem credenciais herdadas, diretório temporário, seed de hash fixa, timeout de 2 segundos e limites Linux de CPU/memória/arquivo. Imports computacionais padrão são permitidos por lista explícita; acesso a arquivos, rede e subprocessos é bloqueado por audit hook. Builtins de I/O, execução dinâmica e reflexão são removidos; atributos privados são bloqueados. Restrições recusadas recebem `code_policy_violation`, não um diagnóstico subjetivo. Essa sandbox simples atende aos outputs locais revisados; não é uma fronteira de segurança contra código hostil arbitrário e não equivale a um container/VM. A execução requer Linux com `resource`.

A avaliação de código comprova os casos finitos executados, não equivalência formal para todo o domínio. Os hashes dos três runs, do benchmark, do registro de hardware, dos módulos e do entrypoint permitem rastrear a avaliação; nenhuma data corrente ou duração do avaliador entra nos artefatos.
