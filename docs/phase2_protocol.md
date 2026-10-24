# Fase 2 — protocolo experimental proposto, versão 1

## 1. Objetivo e escopo

Medir o efeito da disponibilidade de ferramentas locais sobre a acurácia das mesmas dez tarefas do Benchmark V1. O modelo interpreta o problema, seleciona ferramentas, fornece argumentos, recebe resultados e decide quando responder. Acerto da tarefa e eficiência de seleção são medidas distintas. Este documento especifica a implementação futura; não constitui um runner nem altera o dataset.

| Fase | Condição experimental |
|---|---|
| 1 | RAW: sem ferramentas e sem prompting otimizado; concluída |
| 2 | Ferramentas com instruções operacionais mínimas |
| 3 | Ferramentas com prompting deliberadamente otimizado |
| 4 | Fine-tuning do 1B, ferramentas e melhor prompt selecionado na Fase 3 |

Modelos: `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` e `meta-llama/Llama-3.1-8B-Instruct`. Identificação futura: `phase="phase2"`, `condition="tools"`. Tarefas e gabaritos continuam exclusivamente em `prompts/benchmark_v1.json`. Não criar variantes de tarefas fora desse registro.

O [catálogo](phase2_tool_catalog.md) define 25 ferramentas com sobreposição funcional. O catálogo público contém apenas nomes, descrições, argumentos e resultados. Classes, matriz de eficiência, fixtures, gabaritos, testes oficiais e notas de avaliação são privados ao executor/avaliador.

## 2. Hipóteses pré-especificadas

- **H1:** disponibilizar ferramentas aumenta a acurácia em tarefas determinísticas. Comparar diferenças pareadas de acerto por tarefa entre Fases 1 e 2.
- **H2:** o ganho pode ser maior nos modelos menores. Reportar diferença absoluta em pontos percentuais; ganho relativo percentual fica indefinido quando a acurácia RAW for zero, como no 1B observado.
- **H3:** modelos maiores podem selecionar/orquestrar ferramentas melhor sem prompting otimizado. Examinar seleção válida, sequência, erros e duplicatas separadamente do acerto final.
- **H4:** uma estratégia ineficiente pode produzir resposta correta. Cruzar `task_correct`, a métrica principal de sucesso funcional da Fase 2, com a classificação de eficiência.
- **H5:** acurácia de tarefa e eficiência de seleção representam capacidades distintas. Reportar ambas, sem score composto.

São perguntas experimentais, não conclusões. Dez tarefas e um run por modelo permitem descrição pareada; não justificam superioridade global, causalidade isolada de tamanho nem inferência estatística robusta.

## 3. Desenho e comparabilidade

Preservar float16, `device_map="auto"`, `max_new_tokens=512` **por turno**, `do_sample=False`, `temperature=None`, `seed=None`, exatamente dois warm-ups curtos fora do benchmark, descartados. Carregar tokenizer/modelo explicitamente; sincronizar CUDA no dispositivo usado após warm-up e nos limites da medição de geração. Não introduzir quantização, fallback de dtype/CPU, paralelismo ou autocorreção. `device_map=auto` pode provocar offload; registrar mapa efetivo e inviabilidade, sem ocultá-los.

Executar T1–T10 sequencialmente para cada modelo, sem compartilhar histórico, estado transacional ou cache de respostas entre tarefas/modelos. Fixar versão de Python, torch, Transformers, Accelerate, tokenizer, pesos/revision, template de chat, catálogo, schemas, protocolo, fixtures e código de avaliação por hashes no manifesto. Registrar GPU/VRAM, dispositivo, dtype e offload. Carregamento é medido uma vez por run.

O teto proposto de contexto serializado é **16384 tokens**, reservando 512 para cada nova saída: entrada máxima 15872. Mesma regra para os três modelos. Se o contexto exceder o teto ou a capacidade real do modelo, encerrar com `model_error` e detalhe `context_limit`; não descartar mensagens, resumir histórico nem truncar ferramentas. Esse teto deve ser validado por contagem de tokens antes de executar o run oficial. Qualquer revisão do teto após essa verificação deve gerar nova versão do protocolo antes de coletar resultados.

Mudanças inevitáveis frente à Fase 1: catálogo ocupa contexto; há múltiplas inferências, resultados de ferramentas e tokens acumulados. O orçamento de saída por turno permanece 512, mas o orçamento total aumenta até nove turnos (4608 tokens de saída). Portanto, diferenças de acurácia medem o pacote ferramentas + interação + contexto adicional, sem isolar causalmente cada componente. O envelope JSON também consome parte dos 512 tokens de saída; registrar essa diferença, especialmente para código longo, sem ampliar o teto apenas em tarefas específicas.

1B/3B RAW foram executados em T4 e 8B RAW em L4. Registrar novamente o hardware da Fase 2; latência bruta entre fases ou modelos não é comparação controlada. O registro `reports/phase1/hardware/8b_t4_fp16_attempt.md` continua evidência de inviabilidade observada, não um run de acurácia.

## 4. Construção do contexto e neutralidade

### 4.1 Conteúdo permitido

T1–T4 e T7–T10: texto `prompt` completo e inalterado. T5/T6: texto `tools_prompt` completo e inalterado; nunca `prompt`, `raw_prompt` ou o registro inteiro. São as variantes TOOLS já congeladas, com políticas de negócio e resultado final próprios. O catálogo global público é idêntico nas dez tarefas, em ordem lexicográfica de nome, sem destacar ferramentas por tarefa. Não filtrar o catálogo por ID. Os catálogos locais já embutidos em tools_prompt de T5/T6 permanecem no texto congelado; seus nomes/schemas devem coincidir com as definições globais, sem criar ferramentas duplicadas. Uma divergência bloqueia a preparação do run, em vez de autorizar substituição silenciosa.

Acrescentar apenas o catálogo público e o seguinte texto operacional, fixo entre modelos/tarefas:

> Cada saída deve ser um objeto JSON completo, sem texto externo. Há dois tipos de mensagem: tool_call, com exatamente type, tool (string) e arguments (objeto); final_answer, com exatamente type e answer. Em final_answer, answer contém a resposta no formato solicitado pela tarefa; código e texto são strings JSON. A instrução de formato da tarefa aplica-se a answer. Uma mensagem tool_call solicita uma ferramenta do catálogo; seu retorno será acrescentado ao histórico antes da próxima saída. O limite é de 8 chamadas despachadas e 9 saídas do modelo por tarefa. A saída após a oitava chamada pode ser uma resposta final.

Este texto define somente transporte, precedência de formato e limites. Sem exemplos resolvidos ou exemplos de chamadas. Nomes, descrições, schemas e restrições técnicas do catálogo complementam o texto. Descrições não incluem IDs de tarefas, recomendações de seleção ou classes offline.

### 4.2 Serialização proposta para o provider existente

O provider atual recebe uma string e a envolve em uma mensagem `user` pelo chat template. Para preservar essa arquitetura, o futuro adaptador monta um único documento JSON textual com campos ordenados `protocol`, `tool_catalog`, `task`, `history`; `task` contém somente o texto permitido. `history` é uma lista ordenada dos objetos emitidos pelo modelo e dos retornos correspondentes, sem instruções inseridas pelo executor. Aplicar sempre a mesma serialização UTF-8, indentação e ordem de campos, sem dupla aplicação de chat template. Registrar a string exata e sua contagem real de tokens em cada inferência.

A memória do loop é esse histórico explícito, reapresentado integralmente a cada turno. A ferramenta não envia mensagens de sistema nem altera o texto da tarefa. Outputs de ferramentas são dados, nunca novas instruções do executor. Não se pressupõem APIs proprietárias ou suporte nativo a roles de tool calling.

### 4.3 Fronteira com a Fase 3

Fase 2 permite somente o mínimo operacional descrito acima. Pedidos para pensar passo a passo, verificar, autocorrigir, decompor, escolher ferramentas especializadas, aprender com exemplos ou usar um roteiro pertencem à Fase 3. As instruções já existentes em T5/T6 (uma chamada por vez, aguardar retorno e respeitar políticas) são parte do benchmark congelado, não prompting adicional. Essa diferença intrínseca frente às outras oito tarefas deve constar da análise por categoria.

## 5. Protocolo JSON

Todos os objetos têm campos obrigatórios e `additionalProperties=false`. Ordem de chaves é irrelevante. Proibidos chaves duplicadas, NaN/Infinity, coerção de tipos, bool como inteiro, múltiplos objetos, fences e prosa externa. Espaço JSON externo normal é permitido; não completar documentos incompletos.

| Mensagem | Campos exatos e tipos |
|---|---|
| Modelo: chamada | `type`: literal `tool_call`; `tool`: string não vazia; `arguments`: objeto |
| Modelo: final | `type`: literal `final_answer`; `answer`: JSON; tipo/formato final avaliados pelo contrato da tarefa |
| Retorno de execução bem-sucedida | `ok`: literal true; `result`: JSON definido pelo catálogo |
| Retorno de execução recusada/falha | `ok`: literal false; `error`: objeto com `code` e `message`, strings |

São schemas conceituais, não exemplos de resolução. IDs de chamada, hashes, duração, classe da ferramenta e detalhes internos de sandbox ficam no trace privado, fora do retorno público. Associar cada retorno à única chamada pendente pela ordem; não há chamadas paralelas.

**Falha de negócio versus falha da ferramenta:** os `result` dos fixtures são preservados integralmente dentro do envelope. Uma cobrança executada que retorna um resultado de negócio negativo recebe `ok=true` externo e o objeto original em `result`. Uma violação de política ou precondição do dispatch recebe `ok=false` externo, com o código original em `error.code`. Nunca achatar os dois níveis de `ok`. Essa regra é uniforme e deve estar descrita no schema público do retorno, sem revelar qual operação falhará.

Erros públicos usam mensagens fixas por código, sem stack trace, resultados esperados, fixture IDs, caminhos locais ou correções sugeridas. Códigos: `unknown_tool`, `invalid_arguments`, `policy_violation`, `duplicate_call`, `precondition_failed` (domínio); `invalid_expression`, `invalid_data`, `infeasible`, `resource_limit`, `tool_timeout`, `execution_error` (computação). O schema exato de cada erro deve ser congelado antes da implementação oficial; não revelar listas privadas de argumentos válidos.

## 6. Loop, orçamento e encerramento

1. Resetar tarefa e montar contexto permitido; iniciar relógio end-to-end antes da montagem/tokenização do primeiro turno.
2. Gerar uma saída e registrar `BenchmarkResult` do turno. Erro do provider encerra com `model_error` (ou `timeout` quando houver expiração de prazo do provider); falha inesperada do controlador encerra com `executor_error`. Aplicar a classificação técnica da seção 8.1.
3. Parsear a saída inteira. Envelope inválido encerra com `invalid_protocol`, sem repair ou extração.
4. Em `final_answer`, encerrar e avaliar `answer`, inclusive se estiver incorreto. Não solicitar nova tentativa ao modelo.
5. Em `tool_call`, registrar a solicitação. Se já houve oito despachos, encerrar com `tool_limit` sem despachar a nona. Caso contrário consumir uma vaga, validar nome/argumentos e executar ou retornar erro estruturado. Acrescentar chamada e retorno ao histórico e voltar ao passo 2.
6. Timeout global encerra com `timeout`, com escopo/origem e classificação técnica conforme seção 8.1; cancela trabalho pendente e impede commit transacional posterior. Nunca começar a próxima tarefa com processo filho ainda ativo.

Limites propostos uniformes: oito despachos, nove saídas do modelo, **2 s de wall time por ferramenta**, **600 s end-to-end por tarefa**, 256 MiB por subprocesso de código, 8192 bytes UTF-8 para cada retorno serializado. Retorno acima do limite gera `resource_limit`, sem truncamento silencioso. Para ações transacionais, validar serialização/tamanho antes de confirmar alteração de estado. Prazos são limites de viabilidade dependentes de hardware; registrar falhas censuradas e não tratá-las como latência de conclusão.

Nome inexistente ou argumentos inválidos dentro de envelope válido consomem uma vaga e devolvem erro; o modelo pode escolher outra chamada. Envelope cujo `arguments` nem é objeto é `invalid_protocol`. Chamadas repetidas são permitidas e contadas: ferramentas puras retornam novamente, enquanto ferramentas de domínio seguem `duplicate_call` do fixture. O executor nunca escolhe, renomeia ou corrige uma chamada.

**Por que oito?** T5 exige quatro operações de domínio e T6 cinco; restam quatro e três vagas para consultas alternativas, validação ou recuperação. T1 admite recorrência inteira em uma chamada Decimal ou Python; T2 uma busca inteira; T3/T4 um pipeline; T7/T8 uma validação com casos próprios; T9/T10 um processamento geral/validação, ou resposta direta. O teto comporta soluções assistidas sem exigir a solução mínima e limita loops/custo. Não garante sucesso de toda decomposição. Publicar incidência de `tool_limit`; se o teto se mostrar inadequado, revisar o protocolo para outro experimento, sem elevar o orçamento seletivamente durante a coleta.

## 7. Estado e isolamento

Computação/validação: função pura dos argumentos; processo e diretório temporário novos por execução; sem persistência entre chamadas, arquivos externos, rede, relógio ou aleatoriedade fornecidos ao modelo. Modelo deve reenviar explicitamente dados/código de que precisar. Respostas não carregam referências privadas do repositório.

Domínio: snapshot privado por `(run_id, model_id, task_id, attempt_id)`, com conjunto de casos consumidos e log de ações append-only. Copiar `tool_fixtures` do dataset no controlador confiável; aplicar as `dispatch_rules` na ordem original, inclusive igualdade exata de argumentos, bloqueios e `requires`. Resultado negativo de negócio consome o caso; erro de dispatch não consome. Só confirmar uma ação após validação e antes do retorno bem-sucedido, atomicamente; registrar hash de estado antes/depois. Queries extras são somente leitura e não consomem casos nem satisfazem precondições das operações obrigatórias.

Não enviar fixtures, conjunto de casos, IDs de `requires`, gabaritos ou estado completo ao modelo. Cada chamada revela somente a projeção contratada. Encerramento descarta todo estado, depois de salvar trace privado; uma nova tentativa usa novo snapshot. Não repetir automaticamente ações após timeout, nem retomar execução parcialmente consumida como se fosse nova.

Ferramentas gerais recebem somente argumentos fornecidos pelo modelo: não podem acessar o controlador, chamar ferramentas de domínio por dentro, adivinhar snapshots por arquivos ou executar transações. Isso evita que `python_executor` se torne um agente oculto que seleciona ferramentas.

## 8. Sucesso técnico, acerto funcional e contrato congelado

### 8.1 Sucesso técnico e classificação de encerramentos

`technical_success` mantém o significado conceitual da Fase 1: a execução de geração/modelo/provider funcionou tecnicamente, sem falha real de infraestrutura. Não mede obediência do modelo ao envelope, formato, correção da resposta nem eficiência. Preservar o status original de cada `BenchmarkResult`; o parser nunca transforma uma geração com status success em erro do provider.

No episódio, `technical_success=true` exige pelo menos uma geração concluída normalmente, nenhuma geração tentada com erro/interrupção e nenhuma falha real de infraestrutura do executor. Uma falha técnica observada permanece registrada mesmo que uma recuperação permitida possibilite concluir o episódio. Não reclassificar retrospectivamente uma falha técnica como sucesso por causa do resultado final. Se nenhuma geração puder ser concluída, registrar false e o motivo. Encerramento e acerto são campos independentes desse indicador.

| Ocorrência | `technical_success` do episódio | Registro e consequência |
|---|---|---|
| Modelo gera normalmente um envelope JSON inválido | true, se não houve falha técnica anterior | `protocol_correct=false`, `termination_reason=invalid_protocol`; tarefa incorreta, sem reparar JSON |
| Modelo gera resposta final válida no envelope, mas errada no conteúdo/formato | true, se execução técnica íntegra | `termination_reason=final_answer`; avaliar os campos da resposta separadamente |
| Nona solicitação de ferramenta após oito despachos | true, se gerações concluíram sem falha técnica | `termination_reason=tool_limit`; sem resposta final, tarefa incorreta; o envelope pode ser válido |
| Nome/argumentos inválidos, violação de política, precondição ou falha de negócio prevista | Não muda por esse motivo | Erro de chamada/resultado de negócio no trace; retorno estruturado ao modelo quando previsto; recuperação possível |
| Timeout de ferramenta por código do modelo exceder o limite contratado | Não muda por esse motivo, se o isolamento/cancelamento funcionou | Envelope ok=false, `error.code=tool_timeout`, duração e origem `tool_budget`; consumir a vaga e continuar, se restar orçamento global |
| Código do modelo lança exceção dentro da sandbox e ela é capturada conforme contrato | Não muda por esse motivo | `execution_error` de origem `submitted_code`, sanitizado, no trace; não é exceção do controlador |
| Prazo global vence entre operações, sem interromper geração | Pode ser true conforme a regra de integridade acima | `termination_reason=timeout`, `timeout_scope=episode`, `error_origin=episode_budget`; tarefa sem final é incorreta, não falha de protocolo |
| Prazo global interrompe uma geração ainda não concluída | false | `termination_reason=timeout`, `timeout_scope=episode`, `error_origin=episode_budget`, detalhe `generation_interrupted`; é geração não concluída, sem atribuir JSON parcial inválido ao modelo |
| Erro/OOM do provider, timeout próprio do provider ou impossibilidade de iniciar geração | false | `termination_reason=model_error` para erro/OOM/impossibilidade de gerar; `termination_reason=timeout` e `timeout_scope=provider` para prazo próprio expirado; guardar origem provider e código técnico específico |
| Exceção inesperada do dispatcher/controlador, processo de ferramenta indisponível, estado inconsistente ou cancelamento que falhou | false | `termination_reason=executor_error` se impedir continuação; `error_origin=executor_infrastructure`; preservar causa, sem atribuir ao modelo |

Distinguir `tool_timeout` de orçamento aplicado corretamente de timeout causado por infraestrutura indisponível: este último é falha técnica, mesmo que ambos tenham envelope público sanitizado de erro. O trace privado deve registrar origem, escopo, estágio, código e se a operação foi cancelada/confirmada. `resource_limit` aplicado corretamente ao código fornecido pelo modelo não equivale a OOM do provider/controlador. Erro inesperado nunca deve ser disfarçado como falha de negócio.

`termination_reason` aceita `final_answer`, `tool_limit`, `timeout`, `invalid_protocol`, `model_error` e **`executor_error`**. A inclusão do último valor distingue falha do controlador de falha do provider/modelo. Guardar a causa que encerrou o loop e uma lista de eventos anteriores; por exemplo, um erro de ferramenta recuperado não substitui o motivo final. Se o prazo global vencer durante uma ferramenta, não devolver resultado tardio nem continuar o loop; confirmar cancelamento e preservar a evidência de estado. Falha de cancelamento é infraestrutura e torna technical_success=false.

### 8.2 Métricas de resposta e protocolo

O transporte JSON é avaliado separadamente em `protocol_correct`: todas as mensagens emitidas e concluídas pelo modelo devem ser objetos completos válidos de tool_call/final_answer conforme o schema do envelope, sem extração ou repair. Ferramenta desconhecida ou schema de argumentos da ferramenta inválido não tornam o envelope inválido se tool for string e arguments objeto; entram nas métricas de chamadas. O tipo/conteúdo de answer é avaliado no contrato da tarefa, não no parser do envelope. Ausência de resposta final por limite não torna automaticamente o protocolo inválido. Se não houver mensagem concluída, protocol_correct=null (não observado); geração interrompida não é mensagem protocolar concluída.

T1–T4/T9: `answer` é objeto/lista nativo, sem JSON duplamente codificado. T7/T8/T10: `answer` é string, extraída sem strip/reparação. Para JSON, preservar os lexemas numéricos com Decimal, inclusive números exponenciais, sem passar por float. O envelope de transporte é adaptação operacional explícita; não substitui as exigências de formato aplicáveis a answer.

- `format_correct`: answer respeita o formato final exigido pela tarefa, incluindo tipos, campos, ausência de Markdown e regras textuais. Não julga escolha de ferramentas nem ordem do trace.
- `content_correct`: conteúdo lógico/resultado final está correto segundo o gabarito e os testes congelados. Mantém somente o diagnóstico secundário já permitido na Fase 1 (por exemplo, remoção de um fence completo), sem conceder automaticamente crédito funcional ou estrito. Não prova execução de ações de domínio.
- Sem final utilizável, format_correct e content_correct ficam false no placar da tarefa; saídas parciais permanecem apenas como diagnóstico, sem reconstrução de answer.

Para código/texto usar as mesmas regras/casos de avaliação, sem enviá-los ao modelo. Não converter formato inválido em válido por serialização corretiva do executor. Valores corretos mencionados em JSON quebrado não são uma resposta final utilizável.

### 8.3 Métrica principal: task_correct

`task_correct` é a métrica principal de sucesso funcional da Fase 2. Exige encerramento por final utilizável, protocol_correct=true, format_correct=true, content_correct=true e cumprimento efetivo do contrato de negócio. T1–T4/T7–T10 permitem resposta direta correta com zero ferramentas. T5/T6 exigem consultas/ações reais obrigatórias confirmadas no estado e nos retornos, com suas dependências e políticas de negócio; texto correto por adivinhação não basta.

Não exigir o trace integral exato como condição de task_correct. Ferramenta geral, consulta auxiliar permitida ou estratégia mais cara não anulam uma solução funcional correta. Violações que efetivamente mudem o desfecho ou descumpram o contrato de negócio impedem task_correct; simples ineficiência ou erros de chamadas recusadas, seguidos de recuperação sem efeitos indevidos, permanecem diagnósticos separados. Recuperação nunca autoriza repetir operações proibidas ou ignorar efeitos já confirmados.

A classificação de eficiência e os contadores de chamadas **nunca entram na fórmula de task_correct**. O orçamento limita a execução, mas a etiqueta de eficiência não altera retrospectivamente o acerto. technical_success é reportado independentemente: se uma falha técnica recuperável ficar registrada, mas a execução terminar com final válido e solução/ações verificadas, task_correct pode ser true com technical_success=false. Encerramentos sem final por erro, timeout ou limite não recebem crédito funcional mesmo que o estado parcial pareça próximo do objetivo.

### 8.4 Conformidade congelada: strict_correct

`strict_correct` preserva a exigência de conformidade integral com o contrato aplicável do Benchmark V1; não é renomeado nem usado como sinônimo de task_correct. Inclui technical_success=true, protocol_correct=true, formato/conteúdo finais corretos e todas as condições estritas congeladas. Em T1–T4/T7–T10, reutilizar as condições existentes sobre answer. Eficiência não acrescenta restrições a esse contrato.

Em T5/T6, `strict_correct` exige trace completo exatamente igual a `expected_tool_sequence`, argumentos e resultados dos fixtures, resposta `final_expected_answer`, ações confirmadas e zero chamadas extras, conforme `evaluation_notes_tools`. Não basta resolver o negócio. **`dataset_tools_contract_correct` explicita esse mesmo critério integral para T5/T6**, incluindo os requisitos técnico/protocolar e finais; nessas duas tarefas, deve coincidir com strict_correct. Nas outras oito tarefas é null (não aplicável).

Assim, strict_correct=true implica task_correct=true. O inverso não é garantido. content_correct=true isolado pode coexistir com formato inválido ou ausência de ações; format_correct=true isolado não garante conteúdo correto. protocol_correct=true valida o transporte, não o sucesso técnico, a resposta final nem a solução do negócio.

Exemplo abstrato, exclusivamente de avaliação offline, sem gabaritos: uma solução pode ter task_correct=true, strict_correct=false, dataset_tools_contract_correct=false e tool_selection_class=acceptable quando resultado e ações obrigatórias estão corretos, mas uma consulta auxiliar permitida viola o trace estrito congelado. Esse exemplo e estas definições privadas não são instruções adicionais ao modelo.

Publicar task_correct como placar principal, strict_correct como conformidade estrita e dataset_tools_contract_correct para o subconjunto aplicável, sempre com a versão da política. Reportar também taxas de sucesso técnico, protocolo, formato e conteúdo separadamente. Usar todas as tarefas previstas como denominador das taxas principais; null de campo não aplicável/não observado não é acerto e deve ter contagem própria. A taxa de dataset_tools_contract_correct usa somente T5/T6 como denominador (duas tarefas), sem tratar as outras oito como erros. O contrato TOOLS usa somente os fixtures/gabaritos privados no avaliador, nunca no contexto do modelo.

### 8.5 Comparação com a Fase 1

Preservar todos os resultados e definições publicados da Fase 1. Comparar a resolução final nas oito tarefas com formato/alvo mantidos, explicitando o envelope e o loop adicionais. Não renomear o strict_correct RAW publicado como task_correct retroativamente. Para T5/T6, RAW avalia plano com retornos embutidos e TOOLS avalia execução/final: apresentar plano RAW, task_correct TOOLS e contrato integral separadamente. Uma diferença agregada sobre dez tarefas inclui essa mudança de observação; não interpretá-la como ganho puro de ferramentas.

## 9. Métricas de ferramentas

Trace privado por solicitação: ordinal, nome/argumentos exatos, serialização canônica, estado antes/depois, despachada ou bloqueada, envelope de retorno, tempos, decisão subsequente e referência ao turno. Nunca publicar testes privados em traces para treinamento.

| Métrica | Regra determinística |
|---|---|
| `tool_calls_total` | Todas as mensagens tool_call válidas no envelope, incluindo a nona bloqueada; máximo 9 |
| `tool_calls_dispatched` | Vagas consumidas, inclusive erro de validação; máximo 8 |
| `tool_calls_successful` / `tool_calls_failed` | Despachos com envelope externo ok=true / ok=false; soma igual a dispatched; falha de negócio interna não é failed |
| `duplicate_tool_calls` | Solicitações após a primeira com mesmo nome e argumentos canônicos; ordem de objetos ignorada, arrays/tipos preservados; sem tentar inferir equivalência de dois programas diferentes |
| `invalid_tool_calls` | Despachos com unknown_tool ou invalid_arguments; envelope inválido é contado separadamente em invalid_protocol |
| `tool_switches` | Número de nomes diferentes entre solicitações tool_call consecutivas, incluindo inválidas; zero se menos de duas; mudança de camada offline é métrica adicional distinta |
| `tool_selection_valid` | Existe chamada despachada, schema válido e ferramenta tecnicamente capaz conforme matriz offline; mede capacidade potencial, não prova utilidade ou correção dos argumentos semânticos |
| `tool_sequence_correct` | T5/T6: trace integral exato do dataset; outras: null, pois não existe sequência única obrigatória |
| `unnecessary_tool_calls` | Contar somente duplicatas de chamada pura bem-sucedida cujo retorno seja idêntico sem alteração de entradas/estado, ações bloqueadas e duplicatas de domínio recusadas; união sem contar duas vezes. Outras chamadas têm utilidade indeterminada; não classificar todas as gerais como desnecessárias |

Reportar também falhas de negócio, erros de política/precondição e chamadas não despachadas por limite. `unnecessary_tool_calls` é um limite inferior operacional, não julgamento sobre raciocínio interno. Duplicatas recusadas não informam que todo tipo de repetição é errado; uma nova chamada após precondition_failed pode ser necessária e não entra automaticamente como desnecessária.

`hit_max_new_tokens` por turno usa igualdade de tokens de saída com 512. Por tarefa: any hit, número de turnos atingidos e fração. `truncated=null` onde houver hit sem confirmação semântica; false indica somente ausência do indício. Não reparar JSON quebrado após atingir o limite.

## 10. Eficiência offline

A matriz abaixo é privada ao avaliador/desenvolvimento. Conjuntos são de **nomes**, sem ordenação implícita. São alternativas assistidas mínimas; não são planos enviados ao modelo. `minimum_calls` refere-se a uma rota assistida completa, não ao mínimo de toda resposta possível. T1–T4/T7–T10 permitem resposta direta com zero chamadas: `tool_selection_class=null`, `calls_over_minimum=null`, `direct_answer=true`; não recebem invalid por dispensar ferramentas. Em T5/T6, consulta/ação real é parte do contrato.

Classificar rotas assistidas pela primeira regra aplicável:

1. `invalid`: há ferramenta desconhecida, argumentos inválidos, violação de política/precondição ou zero seleção válida; incluir os erros mesmo se houver recuperação.
2. `optimal`: conjunto usado igual a um optimal_tool_set, despachos iguais ao mínimo daquela alternativa, sem duplicatas/erros/retornos excedidos e, no domínio, operações obrigatórias em ordem. Não depende de acertar o conteúdo final.
3. `acceptable`: conjunto contém integralmente uma alternativa optimal ou acceptable e todos os nomes extras estão entre as ferramentas capazes da tarefa; não há chamadas comprovadamente desnecessárias nem falhas. Admite validações adicionais.
4. `inefficient`: restante das rotas assistidas válidas, incluindo repetição pura comprovada ou ferramentas sem capacidade naquela tarefa; não reduz `task_correct` por essa classificação.

Para quem usa ferramentas, `calls_over_minimum=max(0, tool_calls_total-minimum_calls)`; reportar também déficit se total<minimum_calls. O campo não mede custo monetário nem prova que uma chamada extra era dispensável. Conjuntos e classificação devem ter versão/hash antes dos runs; não ajustá-los para favorecer traces observados.

Abreviações apenas nesta matriz: P=python_executor; D=decimal_calculator; B=calculator_basic; I=integer_optimizer; C=csv_query; J=json_query; T=table_transform; S=python_syntax_check; R=python_test_runner; V=json_validator; F=text_format_validator. O catálogo público usa nomes completos. Em T5, A5={lookup_order,check_return,create_return}; em T6, A6={get_stock,reserve_stock,charge,release_stock,log_failure}. Repetições necessárias do mesmo nome contam no mínimo, não no conjunto.

| Tarefa | Ferramentas capazes de ajudar | `optimal_tool_sets` | `acceptable_tool_sets` | `minimum_calls` assistido | Resposta direta / necessidade | Métricas específicas |
|---|---|---|---|---|---|---|
| T1 | D, B, P, V | {D} | {P}; {B}; {D,V} | 1 (D/P); B apenas apoio parcial | Direta possível; ferramenta melhora exatidão | Decimal, arredondamento por etapa, limite de chamadas |
| T2 | I, B, P, V | {I} | {P}; {I,V} | 1 | Direta possível | Restrições, ótimo/desempate, combinações visitadas |
| T3 | C, T, J, P, V | {C} | {T}; {J}; {P}; {C,V} | 1, com dados convertidos pelo modelo quando necessário | Direta possível | latest-before-filter, agrupamento, ordem |
| T4 | J, T, P, V | {J} | {T}; {P}; {J,V} | 1, T exige tabularização explícita pelo modelo | Direta possível | Join de depósitos, campos derivados, ordenação |
| T5 | A5; order_workflow_query; P; V | A5 | A5∪{order_workflow_query}; A5∪{P}; A5∪{V} | 4 no núcleo; 5 em cada alternativa extra | Ações/consultas obrigatórias; P não substitui ações | Sequência de domínio, elegibilidade recebida, confirmação, extras |
| T6 | A6; inventory_query; P; B; V | A6 | A6∪{inventory_query}; A6∪{P}; A6∪{B}; A6∪{V} | 5 no núcleo; 6 em cada alternativa extra | Ações/consultas obrigatórias; compensação real | Reserva, falha de negócio, liberação/log, cobrança não repetida |
| T7 | P, S, R | {R} | {S}; {P}; {S,R} | 1 | Direta possível; casos do modelo só validam hipóteses | Compilação, mutação, cobertura dos casos próprios, erros |
| T8 | P, S, R | {R} | {S}; {P}; {S,R} | 1 | Direta possível | Ordem, sinais, ASCII, exceções e cobertura própria |
| T9 | P, J, T, V | {P} | {J}; {T}; {V}; {P,V} | 1; J/T dependem de extração explícita pelo modelo | Direta possível | Revisões, cancelamentos, tipos, ordenação |
| T10 | P, J, T, F | {P} | {J}; {T}; {F}; {P,F} | 1; J/T dependem de extração explícita pelo modelo | Direta possível | Oito linhas, prefixes, espaços, revisão/severidade |

Algumas ferramentas só contribuem parcialmente (B em T1; S em código; V/F em estrutura), não resolvem sozinhas a tarefa. `acceptable_tool_sets` pode representar ajuda parcial; não certifica suficiência ou acerto. Nenhuma ferramenta especializada de NLP é criada para sintetizar T9/T10: o modelo faz a extração; ferramentas gerais/tabulares auxiliam com os dados que ele fornecer. B não tem arredondamento monetário por etapa: é aceitável como apoio parcial, deixando arredondamentos ao modelo. Para classificação/`calls_over_minimum` de T1, o baseline assistido único é 1, não um mínimo matemático demonstrado para toda estratégia. Não inferir um ótimo global de custo a partir desse baseline.

## 11. Latência, tokens e custo

Guardar todos os `BenchmarkResult` de inferência por turno sem alterar sua dataclass. Um futuro registro de episódio os referencia e soma input/output/total_tokens, tempo de inferência e métricas de ferramenta. Tokens de entrada contam o histórico reapresentado **a cada chamada**, sem deduplicação. Tempo end-to-end inclui montagem/tokenização, inferência, parse, despacho e ferramentas, excluindo carga/warm-up e avaliação offline. Guardar duração de cada ferramenta e soma; não somar novamente carga de modelo por tarefa/turno.

Usar relógio monotônico, CUDA sincronizado quando aplicável. Registrar tempos parciais em falhas. Throughput do modelo = outputs acumulados / inferência acumulada; throughput de episódio = outputs / tempo end-to-end, nomeado distintamente. Não usar média simples dos throughputs por turno como agregado. Estimativa futura de custo computacional pode ser campo opcional com método/versionamento; não inventar preço monetário. Execução local sem API paga não implica custo físico nulo de GPU.

## 12. Riscos e critérios de aceite

Riscos: catálogo grande pode consumir contexto; mais turnos ampliam orçamento; schemas/erros podem funcionar como ajuda; validadores com valores esperados vazariam gabarito; testes criados pelo modelo não garantem correção; conjuntos ótimos são convenções limitadas; T5/T6 mudam plano para execução; hardware/timeout censuram desempenho. Prevenir contaminação por fine-tuning excluindo dataset de avaliação, casos privados e traces com respostas/gabaritos dos corpora de treino; controlar proveniência e hashes na Fase 4.

Antes de coletar oficialmente, exigir:

- As dez tarefas e todos os artefatos RAW/Fase 1 com os mesmos hashes; manifesto da Fase 2 separado, sem sobrescritas.
- Parser estrito testado com mensagens incompletas, duplicadas, tipos inválidos e finais de código/texto; nenhum repair.
- Mesmo catálogo/protocolo público entre modelos/tarefas; revisão de conteúdo para excluir matriz, classes, fixtures e casos privados. Python e test runner sem acesso aos mounts/ambiente do controlador.
- Fixtures respeitados byte/estrutura e semântica originais; testes de isolamento, reset, duplicatas, precondições, erros de negócio e compensação; queries extras incapazes de consumir casos ou antecipar elegibilidade/cobrança.
- Testes do teto: oitava chamada seguida de final, nona bloqueada, orçamento consumido por erro, timeout/cancelamento e ausência de efeitos após cancelamento.
- Verificação de recursos e contexto sem modelos reais na primeira implementação; depois smoke test operacional público, separado de V1, sem ajustar estratégia com base em acertos oficiais.
- Avaliação final e de trace independentes, com casos de resposta correta por rota geral/consulta extra; demonstração de que eficiência não altera `task_correct`, de que `strict_correct` preserva o contrato congelado e de que coincide com `dataset_tools_contract_correct` em T5/T6. Cobrir geração normal com envelope inválido (sucesso técnico e falha protocolar), resposta direta correta, ações obrigatórias reais, timeout de ferramenta/episódio e falhas de infraestrutura com origens distintas.
- Reexecução dos traces com mesmas entradas produz mesmos resultados de ferramentas/placares; latências podem variar e nunca entram em decisão de acerto além dos limites pré-fixados.

## 13. Plano posterior

1. Congelar esta especificação, schemas e distinção entre sucesso técnico, task_correct principal e conformidade estrita em revisão metodológica; resolver detalhes marcados no catálogo antes de executar modelos.
2. Implementar parser, catálogo público por allowlist, dispatcher e loop com fakes; runner RAW e avaliador RAW permanecem próprios da Fase 1.
3. Implementar ferramentas puras e sandbox com isolamento de SO adequado ao código não confiável; não reutilizar sem revisão a sandbox simples de avaliação como fronteira de segurança do agente.
4. Implementar ferramentas transacionais sobre fixtures, queries somente leitura e trace/reset; testar equivalência com o dataset.
5. Implementar registro de episódio, agregação, métricas offline e exportação em diretórios novos da Fase 2, preservando resultados por turno.
6. Validar neutralidade, contagem de contexto, limites, testes de isolamento e reprodução; só depois preparar Colab para os três modelos e coletar Fase 2.
7. Congelar dados/avaliação da Fase 2 antes de estudar prompting na Fase 3. Não usar ajustes da Fase 3 retroativamente nos resultados da Fase 2.
