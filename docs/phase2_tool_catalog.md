# Catálogo conceitual da Fase 2 — versão 1

Este arquivo é especificação de desenvolvimento. **Não enviar este documento inteiro ao modelo.** A projeção pública futura contém somente `name`, `description`, `parameters` e `returns`, sem categorias, classes, comentários metodológicos ou correspondência com tarefas. Publicar as 25 ferramentas em ordem lexicográfica, com o mesmo texto em todos os modelos e tarefas. Nenhuma ferramenta contém algoritmo de resolução específico do benchmark.

## 1. Convenções de schema e execução

A notação abaixo equivale a schemas fechados: campos obrigatórios, tipos estritos e nenhum campo extra, salvo explicitamente opcional. `string+` é string não vazia; `int+` inteiro >=1; `JSON` é qualquer valor JSON finito com chaves únicas. Inteiro não inclui booleano. Números decimais de precisão arbitrária usam strings ASCII conforme `-?(0|[1-9][0-9]*)(\.[0-9]+)?`, sem exponentes; essa convenção pertence às **ferramentas**, não muda o contrato final numérico das tarefas. Ordenação usa Unicode lexicográfico para strings e valor numérico para números; comparações não fazem coerção.

Todos os resultados da tabela são o conteúdo de `result` no envelope externo definido em [phase2_protocol.md](phase2_protocol.md). Ferramentas recebem somente os argumentos; não recebem task_id, gabarito, notas ou casos privados. Ferramentas puras não possuem estado persistente. Cotas comuns: 2 s wall time, saída serializada até 8192 bytes; ferramentas de código em processo separado com até 256 MiB. Argumentos completos também estão sujeitos ao teto de contexto; pipelines e expressões têm limites explícitos abaixo.

Um erro de schema gera `invalid_arguments`; erro de linguagem/dados gera `invalid_expression` ou `invalid_data`; domínio usa os códigos congelados do dataset. Erros não incluem sugestões de argumentos, valores esperados ou trechos de gabarito. Mensagens públicas fixas por código devem ser parte do contrato versionado.

## 2. Catálogo completo

As classes desta tabela existem apenas para análise offline. Descrições públicas futuras devem usar somente a coluna de capacidade, sem as colunas de classe ou comentários de uso.

| Nome | Classe offline | Capacidade / descrição pública proposta | Argumentos fechados | Resultado de sucesso |
|---|---|---|---|---|
| `calculator_basic` | especializada | Avalia aritmética numérica finita, sem acesso a variáveis externas | `expression: string+` | `{value: string decimal}` |
| `decimal_calculator` | especializada | Executa operações Decimal com arredondamento e estado explícitos | `precision: int[28..100]`, `rounding: enum`, `variables: objeto de strings decimais`, `steps: lista de Step`, `outputs: lista de identificadores` | `{values: objeto de strings decimais}` |
| `integer_optimizer` | especializada | Enumera uma busca inteira limitada com restrições lineares e objetivo lexicográfico | `variables: lista de Variable`, `constraints: lista de Constraint`, `objectives: lista não vazia de Objective` | `{assignment: objeto de inteiros, objective_values: lista de inteiros, combinations_examined: inteiro}` |
| `csv_query` | especializada | Converte CSV tipado e aplica pipeline tabular explícito | `csv: string`, `delimiter: string de um caractere`, `columns: objeto nome→tipo`, `pipeline: lista de Stage` | `{rows: lista de objetos}` |
| `json_query` | especializada | Seleciona dados JSON e aplica pipeline com relações e campos derivados | `data: JSON`, `root: Path`, `pipeline: lista de Stage` | `{rows: lista de objetos}` |
| `table_transform` | intermediária | Transforma listas de objetos por pipeline tabular | `rows: lista de objetos`, `pipeline: lista de Stage` | `{rows: lista de objetos}` |
| `python_executor` | geral | Executa código Python restrito sobre dados fornecidos e retorna seu valor JSON | `code: string+`, `data: JSON` | `{value: JSON}` |
| `lookup_order` | especializada | Consulta estado e linhas de um pedido | `order_id: string+` | Objeto original do fixture da operação |
| `check_return` | especializada | Consulta elegibilidade de uma linha e motivo | `line_id: string+`, `reason: string+` | Objeto original com `eligible: boolean` |
| `create_return` | especializada | Cria uma devolução com linhas informadas | `order_id: string+`, `line_ids: lista não vazia de strings+ únicas` | Objeto original com `ok: boolean` |
| `cancel_order` | especializada | Solicita cancelamento de um pedido | `order_id: string+` | Conforme fixture, ou erro de política |
| `refund_payment` | especializada | Solicita estorno de pagamento de um pedido | `order_id: string+` | Conforme fixture, ou erro de política |
| `order_workflow_query` | intermediária | Projeta campos de leitura de um pedido, sem ações | `order_id: string+`, `fields: lista não vazia de enum(status,lines) sem repetição` | `{order: objeto com somente campos solicitados}` |
| `get_stock` | especializada | Consulta quantidade disponível de um SKU | `sku: string+` | Objeto original com `available: inteiro` |
| `reserve_stock` | especializada | Reserva unidades para um pedido | `order_id: string+`, `sku: string+`, `quantity: int+` | Objeto original com `ok` e identificador quando aplicável |
| `charge` | especializada | Cobra um valor em centavos de um pedido | `order_id: string+`, `amount_centavos: int+` | Objeto original com `ok` e erro de negócio quando aplicável |
| `release_stock` | especializada | Libera uma reserva pelo identificador recebido | `reservation_id: string+` | Objeto original com `ok` |
| `log_failure` | especializada | Registra motivo de falha em um pedido | `order_id: string+`, `reason: string+` | Objeto original com `ok` |
| `issue_invoice` | especializada | Emite nota de um pedido | `order_id: string+` | Conforme fixture, ou erro de política |
| `send_receipt` | especializada | Envia recibo de um pedido | `order_id: string+` | Conforme fixture, ou erro de política |
| `inventory_query` | intermediária | Projeta disponibilidade de uma lista de SKUs, sem reservar ou cobrar | `skus: lista não vazia de strings+ únicas` | `{rows: lista de {sku: string, available: inteiro}}`, na ordem solicitada |
| `python_syntax_check` | validação | Compila código Python sem executá-lo | `code: string` | `{valid: boolean, errors: lista de {code: string, line: inteiro ou null, column: inteiro ou null}}` |
| `python_test_runner` | validação | Executa uma função sobre casos fornecidos na chamada | `code: string+`, `function: identificador`, `cases: lista não vazia de TestCase` | `{passed: boolean, cases: lista de {index: inteiro, passed: boolean, actual: JSON ou null, exception: string ou null}}` |
| `json_validator` | validação | Valida texto JSON contra schema estrutural fornecido | `text: string`, `schema: StructuralSchema` | `{valid: boolean, errors: lista de {path: Path, code: string}}` |
| `text_format_validator` | validação | Valida propriedades formais de linhas de texto | `text: string`, `line_count: inteiro>=0`, `prefixes: lista de strings`, `allow_empty_lines: boolean`, `allow_outer_spaces: boolean`, `allow_final_newline: boolean` | `{valid: boolean, errors: lista de {line: inteiro ou null, code: string}}` |

São 3 matemáticas, 3 de dados, 1 geral, 6 de pedidos, 8 de estoque/pagamento e 4 validadores. Não existem aliases nem nomes que resolvam uma tarefa específica. O enquadramento offline por camada não estima custo real: medir duração, não presumir uma constante de preço por classe.

## 3. Linguagens declarativas compartilhadas

### 3.1 Expressões aritméticas

`calculator_basic`: gramática fechada de literais decimais, parênteses, sinais unários e operadores `+`, `-`, `*`, `/`, `//`, `%`. Sem funções, nomes, potência, loops ou eval Python. Até 256 nós e 4096 caracteres por expressão; avaliar com Decimal, precisão fixa 28, arredondamento HALF_EVEN. Divisão por zero e resultado não finito são erros. Não oferece arredondamento monetário por etapa; isso diferencia sua capacidade da ferramenta Decimal explícita. Retorno é representação decimal canônica sem expoente, zero sem sinal e sem zeros fracionários redundantes.

`decimal_calculator`: identificadores ASCII `[A-Za-z][A-Za-z0-9_]*`, sem nomes reservados da linguagem. `rounding` aceita `ROUND_HALF_UP`, `ROUND_HALF_EVEN`, `ROUND_DOWN`, `ROUND_UP`, `ROUND_FLOOR`, `ROUND_CEILING`; vale para todas as operações sujeitas ao contexto Decimal. Operações exatas não recebem arredondamento adicional implícito.

`Step` é uma união fechada:

- atribuição: `{op:"assign", name: identificador, expression: string+}`;
- quantização: `{op:"quantize", name: identificador, expression: string+, quantum: string decimal positiva}`;
- repetição: `{op:"repeat", count: inteiro[1..1000], steps: lista de atribuições/quantizações}`.

Expressões usam a gramática básica mais leitura de variáveis já definidas. `assign` substitui o valor de uma variável; os passos ocorrem na ordem fornecida, com estado local. Repetição não pode conter outra repetição; máximo de 10000 operações expandidas, 100 variáveis e 256 nós por expressão. `outputs` só pode nomear variáveis existentes, sem duplicatas. Isso permite recorrências e arredondamento por etapa, sem embutir cronograma de empréstimo, taxas, valores ou regra de tarefa. Retorno só contém as variáveis solicitadas, como strings; converter para números na resposta final é responsabilidade do modelo, sem coerção pelo runner.

### 3.2 Busca inteira

`Variable={name: identificador, min: inteiro, max: inteiro}`; nomes únicos, min<=max, limites inclusivos obrigatórios. Produto dos tamanhos dos domínios <=1000000; erro `resource_limit` acima disso. Não inferir limites ausentes a partir de gabaritos.

`Constraint={coefficients: objeto nome→inteiro, relation: enum(eq,le,ge), rhs: inteiro}` representa soma linear. `Objective={coefficients: objeto nome→inteiro, direction: enum(min,max)}`. Coeficiente omitido significa zero, nome desconhecido é erro. Avaliar objetivos lexicograficamente na ordem informada: desempates são objetivos adicionais explícitos, nunca heurísticas escondidas. Empate em todos os objetivos: escolher tupla de atribuições lexicograficamente crescente na ordem da lista de variáveis. Enumeração segue essa mesma ordem; `combinations_examined` conta todas as atribuições visitadas, viáveis ou não. Ausência de solução retorna `infeasible`. Não aceita função Python como restrição.

### 3.3 Pipeline tabular e hierárquico

`Path` é uma lista de chaves string e/ou índices inteiros>=0. Lista vazia designa o valor atual. Não usar sintaxe ambígua de ponto, avaliação dinâmica ou curingas. Campo ausente, índice fora do limite e comparação de tipos incompatíveis geram `invalid_data`; não omitir silenciosamente linhas. Pipeline até 32 stages, entrada até 10000 linhas/objetos, intermediário até 100000 linhas. Não interpretar prosa do problema por NLP.

`csv_query` usa CSV padrão com header, quoting padrão, `delimiter` explícito e número de colunas uniforme. `columns` deve cobrir exatamente os headers únicos; tipos aceitos `string`, `integer`, `decimal`, `boolean`. Inteiros ASCII assinados sem fração; booleanos somente `true`/`false`; Decimal conforme convenção acima. Sem inferência automática. Decimal tipado permanece valor Decimal interno, serializado como string canônica; schema público informa essa diferença. `table_transform` recebe JSON tabular já fornecido. `json_query` seleciona `data[root]`, que deve ser lista de objetos; relações referenciam outras listas no próprio `data`.

`Stage` é união discriminada por `op`, com os seguintes campos exatos:

| op | Campos além de op | Semântica |
|---|---|---|
| `latest_by` | `keys: lista de Paths`, `version: Path` | Por chave, manter maior versão numérica; empate gera erro, sem eleger vencedor arbitrário |
| `filter` | `conditions: lista de {path: Path, operator: enum(eq,ne,lt,le,gt,ge,in), value: JSON}` | AND; `in` exige value lista; igualdade tipada |
| `derive` | `name: string+`, `expression: Expr` | Acrescentar/substituir campo com expressão tipada |
| `explode` | `path: Path`, `as: string+` | Uma linha por elemento da lista; copiar os outros campos; lista vazia produz zero linhas |
| `join` | `right: lista de objetos ou objeto {root: Path}`, `left_key: Path`, `right_key: Path`, `as: string+` | Inner join por igualdade tipada; right.root somente em json_query; correspondências na ordem original; resultado right fica no campo as |
| `group` | `keys: lista de {name: string+, path: Path}`, `aggregates: lista de {name: string+, function: enum(sum,count), path: Path ou null}` | count exige path=null; sum exige path numérico; saída contém só chaves e agregados; grupo por igualdade tipada |
| `sort` | `by: lista não vazia de {path: Path, direction: enum(asc,desc)}` | Ordenação lexicográfica estável dos critérios; empates preservam entrada |
| `project` | `fields: lista de {name: string+, path: Path}` | Construir objetos só com os campos nomeados |

Nomes de saída duplicados em um stage são erro. Group mantém a ordem de primeira aparição das chaves até um sort explícito. `latest_by` mantém a ordem da primeira aparição de cada grupo, substituindo o valor na mesma posição.

`Expr` é `{field: Path}`, `{literal: JSON}` ou `{operator: enum(add,subtract,multiply), operands: lista de exatamente duas Expr}`. Até 64 nós; operadores aritméticos exigem números do mesmo tipo interno; tipos mistos geram erro, sem promoção automática ou conversão de strings. Campos JSON numéricos decimais são lidos com Decimal; strings continuam strings. Pipeline não tem loops, ranges, interpretação de Python ou acesso a dados não fornecidos. A ordem dos stages é decidida pelo modelo, não pelo executor.

## 4. Ferramentas gerais e de validação

### 4.1 Python controlado

`python_executor` exige código que defina `main(data)`; executa essa função uma vez e retorna seu valor JSON. Nenhuma impressão, leitura stdin, filesystem externo, sockets, subprocessos, imports de terceiros ou acesso ao repositório/variáveis de ambiente. Não oferece tool dispatch interno. Bibliotecas padrão computacionais permitidas devem ser enumeradas na implementação e publicadas igualmente aos modelos; proposta inicial: re, math, decimal, collections, itertools, functools, operator, statistics, bisect, heapq, string e json. Sem relógio, random, os, pathlib, pickle, ctypes ou acesso reflexivo ao controlador.

A ferramenta valida sintaxe, assinatura e política antes da execução. Não existe acesso implícito ao prompt: o modelo fornece `data` e `code`. Datas/fixtures/gabaritos/testes do controlador nunca são montados no processo. Valores retornados devem ser JSON finito; objetos Python arbitrários são `invalid_data`. Ambiente sanitizado, hash seed fixo (`PYTHONHASHSEED=0`), diretório temporário descartável, isolamento de rede/filesystem por SO, CPU/memória/processos limitados e cancelamento verificável. A sandbox simples de `benchmark/evaluation` é referência de testes offline, não uma garantia suficiente para execução geral de código não confiável.

`python_syntax_check` não executa código. Retorna `valid=false` com localização e código estável para falha sintática; não reescreve o código. Assim, uma verificação executada pode ter envelope externo ok=true e valid=false.

`python_test_runner` usa a mesma fronteira de isolamento, define a função indicada pelo modelo e executa **somente seus casos**. `TestCase` contém exatamente `args: lista JSON`, `kwargs: objeto` e um dos campos `expected: JSON` ou `raises: enum(ValueError,TypeError,KeyError,IndexError,ZeroDivisionError)`. Até 100 casos; limite de 2 s para o conjunto inteiro. Comparar tipos/valores/ordem, sem equivalência bool/int. Cópia profunda por caso, verificar argumentos não mutados e saída stdout vazia. Retornar índice, resultado JSON ou nome de exceção sanitizado; mensagens de exceção não expõem caminhos. Não inventar casos nem consultar os casos públicos de avaliação existentes no registro privado. Somente casos explicitamente transmitidos pelo modelo são executados, ainda que ele derive exemplos do texto do problema.

### 4.2 Validação de formato

`StructuralSchema` aceita somente `type` (object,array,string,integer,number,boolean,null), `properties`, `required`, `additionalProperties` (boolean), `items`, `minItems`, `maxItems`. Limite 16 níveis, 256 nós. Não aceita `const`, `enum`, comparação de valores esperados, referências externas ou palavras-chave desconhecidas. Tipos object/array exigem respectivamente properties/required/additionalProperties e items; tamanhos só se aplicam a arrays. Números são finitos, bool não é number/integer. Schema inválido é erro de argumentos; documento válido que não satisfaz schema retorna valid=false. `json_validator` parseia todo `text`, rejeita duplicatas e conteúdo extra, sem remover fences. Ordem de erros estável por caminho e código, sem fornecer o valor que tornaria a resposta correta.

`text_format_validator`: normalizar CRLF para LF; quando allow_final_newline=true, retirar no máximo um LF final, senão ele conta como linha adicional. `line_count` é número exato de linhas. `prefixes` deve ter exatamente line_count itens e exigir startswith na linha correspondente; não contém conteúdo correto esperado. allow_empty_lines e allow_outer_spaces controlam respectivamente linhas vazias e espaço inicial/final de cada linha. Não avalia severidade, fatos, IDs ou valores da resposta. A ferramenta não recupera nem formata o texto automaticamente.

Validadores só verificam restrições fornecidas na chamada. Se o modelo construir schema/prefixes incorretos, um resultado valid=true não certifica o contrato oficial. O avaliador offline continua independente.

## 5. Queries e ferramentas transacionais

As doze ferramentas originais de domínio preservam nomes, argumentos, precondições, bloqueios e retornos do catálogo/fixtures de T5/T6. A implementação copia essas definições do dataset; não duplica IDs, números ou respostas nesta especificação. Fora dos casos permitidos, retornar exatamente os erros de dispatch previstos. Um catálogo maior não autoriza ações novas no cenário.

`order_workflow_query` é incluída por permitir projeção read-only reutilizável de pedido; não verifica elegibilidade, não planeja fluxo, não devolve próximos passos e não cria devoluções. Só expõe campos do mesmo read model da consulta do pedido, quando solicitada. Não lê `expected_tool_sequence`, estado de casos consumidos ou resultados futuros. Para pedido desconhecido, `invalid_arguments`. A operação obrigatória `lookup_order` continua necessária para satisfazer o contrato de negócio de T5; a query não consome seu caso.

`inventory_query` faz leitura em lote e projeção de disponibilidade dos mesmos read models de estoque; preserva ordem solicitada, não calcula compensação, não revela falha futura de pagamento e não substitui reservas/cobranças. Todos os SKUs devem existir no snapshot; um desconhecido faz falhar a chamada inteira, sem resultado parcial. Não consome `get_stock` nem satisfaz seus requires. Essas queries criam redundância de leitura realista, sem serem botões de resolução; sua utilidade será medida offline.

Para cenários transacionais atuais, somente o mock local/fixture existe: nunca integrar serviços externos, criar reembolsos reais ou emitir notas reais. A disponibilidade no catálogo é constante; a elegibilidade e os bloqueios são determinados pelo estado privado, não por instruções de seleção ao modelo.

## 6. Congelamento antes da implementação oficial

Este catálogo define interfaces conceituais com gramáticas e limites. Antes de coletar modelos, transcrever os tipos para schemas fechados públicos, fixar mensagens de erro e serialização Decimal/Unicode, testar conformance das três ferramentas de pipeline e validar isolamento de SO. Nenhum desses ajustes pode acrescentar exemplos resolvidos, preferência por classe ou gabaritos ao catálogo público.

A sobreposição é funcional: Decimal/Python para recorrências, optimizer/Python para busca, CSV/JSON/tabela/Python para dados, execução/testes/sintaxe para código e queries projetadas para leitura de domínio. Capacidades e restrições distintas justificam cada nome. Não adicionar ferramentas que selecionam a rota ou devolvem uma resposta pronta por task_id.
