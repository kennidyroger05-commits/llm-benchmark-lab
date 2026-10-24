# Benchmark v1 — coleção oficial

Fonte oficial: `benchmark_v1.json` (lista JSON com exatamente 10 tarefas, T1–T10).
Esta documentação é uma visão legível do mesmo conteúdo. Os prompts são autocontidos
e independentes do provider. Não há chamadas de inferência, runner ou modelos nesta entrega.

## Protocolo

- Aplicar a coleção única a Llama 3.2 1B Instruct, Llama 3.2 3B Instruct e Llama 3.1 8B Instruct. Em T5/T6, selecionar a variante experimental documentada abaixo; dentro de cada condição, usar o mesmo prompt para todos os modelos. Os prompts das outras oito tarefas permanecem inalterados. FINE-TUNED continua uma condição futura, sem novo dataset nesta revisão.
- `tool_required` indica se uma ferramenta aumenta materialmente a confiabilidade; não é requisito para responder nem autorização para mudar o prompt. `recommended_tool` é uma capacidade lógica, sem vínculo com API ou provider.
- Em RAW, não disponibilizar ferramentas. Em T5/T6 RAW, os retornos simulados estão embutidos em `raw_prompt`; a resposta é um plano JSON. Em T5/T6 TOOLS, usar `tools_prompt` e ferramentas locais reais com respostas controladas por fixtures determinísticos: chamar, receber resultado e decidir iterativamente. Nenhuma API externa é necessária.
- Para as outras oito tarefas, enviar somente `prompt`. Em T5/T6 RAW, enviar somente `raw_prompt` (`prompt` é seu alias de compatibilidade). Em T5/T6 TOOLS, enviar somente `tools_prompt` e as definições de `tool_catalog` no canal de ferramentas. Nunca enviar o registro completo, `raw_prompt`, gabaritos, fixtures, sequência esperada, resposta final esperada ou notas de avaliação ao modelo em TOOLS. Somente o resultado da chamada efetivamente realizada é entregue, na vez correspondente, pelo ambiente. Não incluir as tarefas oficiais nem suas respostas nos dados de fine-tuning, evitando contaminação.
- Não foram medidas diferenças entre tamanhos de modelos. A dificuldade é uma classificação de projeto; discriminação e calibração dependem de uma execução futura.

## Índice

| ID | Categoria | Título | Ferramenta recomendada | Tipo de avaliação | Dificuldade |
| --- | --- | --- | --- | --- | --- |
| T1 | Cálculo Financeiro / Lógica Matemática Exata | Saldo com amortização extraordinária | python_calculator | numeric | hard |
| T2 | Cálculo Financeiro / Lógica Matemática Exata | Compra inteira de custo mínimo | python_calculator | json_schema | hard |
| T3 | Manipulação e Filtragem de Dados | Versões de pedidos antes do filtro | csv_reader | json_schema | hard |
| T4 | Manipulação e Filtragem de Dados | Estoque disponível em depósitos habilitados | json_filter | json_schema | medium |
| T5 | Intenção e Roteamento de Múltiplas Ferramentas | Devolução parcial por elegibilidade | router | json_schema | medium |
| T6 | Intenção e Roteamento de Múltiplas Ferramentas | Compensação após falha de pagamento | router | json_schema | hard |
| T7 | Programação e Automação | Reconciliação sem mutar eventos | python_executor | code | hard |
| T8 | Programação e Automação | Expansão ordenada de intervalos | python_executor | code | hard |
| T9 | Síntese e Formatação Estruturada Restrita | Ata consolidada com revisões | none | json_schema | hard |
| T10 | Síntese e Formatação Estruturada Restrita | Boletim operacional em oito linhas | none | structured_text | hard |

## Convenções de avaliação

Respostas JSON devem ser documentos JSON válidos, sem cercas Markdown ou prosa.
A ordem das chaves é irrelevante; a ordem de listas é relevante. Inteiros não incluem
booleanos; chaves e campos extras são rejeitados nas respostas estruturadas.
`json_schema` exige formato **e igualdade dos valores** especificados, não apenas
validação de tipos. `numeric` usa comparação decimal exata nas unidades pedidas.
`code` aceita qualquer implementação equivalente ao contrato: o código de referência
não é a única solução correta. Os vetores de teste são exemplos executáveis do contrato,
não substituem a especificação completa. A futura execução de código gerado deve ocorrer
em ambiente isolado. `structured_text` segue as normalizações estritas descritas na tarefa.

## Condições experimentais de T5 e T6

Cada uma mantém um único registro e um `prompt_base` comum: RAW e TOOLS resolvem
o mesmo problema, com a mesma intenção, políticas, catálogo e desfecho de negócio.
RAW recebe os retornos antecipadamente e entrega um plano; TOOLS precisa executar
as chamadas locais, receber resultados e decidir a próxima ação. A resposta final
de TOOLS resume o resultado que o plano RAW correto produz: os formatos de resposta
diferem, mas o resultado de negócio é idêntico. O gabarito RAW original foi preservado.

| Aspecto | RAW | TOOLS |
| --- | --- | --- |
| Entrada do modelo | `raw_prompt`, com catálogo e retornos simulados | `tools_prompt` e `tool_catalog`, sem retornos antecipados |
| Interação | Sem execução; plano de chamadas em JSON | Chamadas locais reais, sequenciais, com resultado após cada chamada |
| Saída avaliada | Lista idêntica a `expected_answer` | Trace idêntico a `expected_tool_sequence` e resposta igual a `final_expected_answer` |
| Dados privados do avaliador | Gabarito e notas | Fixtures, gabaritos e notas; nunca enviar estes campos como contexto |

`tool_catalog` contém nomes, descrições e esquemas JSON dos argumentos, sem valores
de retorno do cenário. `tool_fixtures.cases` associa argumentos exatos a resultados
determinísticos; `requires` referencia casos que precisam ter sido consumidos.
`blocked_tools`, `errors` e `dispatch_rules` definem respostas para chamadas
desnecessárias, inválidas, fora de ordem e repetidas. As regras são dados de contrato
para um executor futuro, não uma implementação de ferramentas ou runner.
Todo catálogo tem cobertura: casos válidos, ferramentas bloqueadas e fallback para
argumentos inválidos. Cada sessão começa com estado novo. Uma resposta de negócio
`ok=false` também consome seu caso, permitindo a compensação subsequente.

`expected_answer` continua sendo o gabarito RAW; `expected_tool_sequence` preserva
as mesmas chamadas. `final_expected_answer` descreve o desfecho comum. Em TOOLS,
avaliar seleção, ordem, argumentos, chamadas extras, dependência dos retornos e
resposta final separadamente; sucesso integral exige todos esses critérios.
O avaliador pode observar uso correto de dados e dependências no trace, sem afirmar
que isso demonstra o raciocínio interno do modelo.

**Isolamento obrigatório:** o conteúdo de gabaritos e fixtures nunca deve ser
enviado como prompt, catálogo ou contexto antecipado em TOOLS. Isso inclui IDs de
reserva, motivos de falha, elegibilidades e quantidades disponíveis. Resultados
individuais são revelados apenas depois da chamada correspondente. Não concatenar
`prompt_base` ao prompt da condição: cada variante já inclui o problema comum.

## Tarefas e gabaritos

### T1 — Saldo com amortização extraordinária

Cálculo Financeiro / Lógica Matemática Exata · hard · `numeric`

**Prompt exato:**

```text
Um empréstimo tem saldo inicial de R$ 12.750,00 e juros mensais de 1,375%.
Em cada um dos meses 1 a 6, nesta ordem:
1. Calcule juros sobre o saldo ao início do mês.
2. Arredonde SOMENTE esses juros para centavos por ROUND_HALF_UP
   (em caso de empate, arredonde para longe de zero).
3. Some os juros arredondados ao saldo e subtraia a prestação de R$ 2.050,00.
4. Apenas no mês 3, subtraia ainda uma amortização extra de R$ 850,00,
   depois da prestação. Não há outros custos nem arredondamentos.
Retorne somente um objeto JSON com saldo_final e juros_totais, ambos
números em reais com duas casas decimais. juros_totais é a soma dos seis
valores de juros já arredondados. Não inclua explicação ou bloco Markdown.
```

**Gabarito determinístico:**

```json
{
  "saldo_final": 222.51,
  "juros_totais": 622.51
}
```

**Critério:** Exigir objeto JSON apenas com saldo_final e juros_totais, números (não strings). Comparar por Decimal com os valores esperados; tolerância zero. Aceitar zeros decimais finais equivalentes. Não arredondar somente ao final. Juros mensais de referência: 175.31, 149.54, 123.40, 85.23, 58.21, 30.82.

### T2 — Compra inteira de custo mínimo

Cálculo Financeiro / Lógica Matemática Exata · hard · `json_schema`

**Prompt exato:**

```text
É necessário comprar exatamente 137 unidades de um insumo. Existem três
tipos de pacote, disponíveis em quantidades inteiras não negativas:
A: 12 unidades por 5890 centavos; B: 7 unidades por 3510 centavos;
C: 1 unidade por 620 centavos. Use no máximo 8 pacotes A, pelo menos
2 pacotes B e no máximo 6 pacotes C. Não compre unidades excedentes.
Minimize o custo total em centavos. Se houver empate, escolha a solução
com menos pacotes C; persistindo o empate, com menos pacotes B.
Retorne somente JSON com as chaves pacotes_a, pacotes_b, pacotes_c,
custo_centavos, todas inteiras. Não inclua Markdown ou explicações.
```

**Gabarito determinístico:**

```json
{
  "pacotes_a": 5,
  "pacotes_b": 11,
  "pacotes_c": 0,
  "custo_centavos": 68060
}
```

**Critério:** Objeto com exatamente quatro campos inteiros, sem booleanos; igualdade integral com o gabarito. Validar 12*A+7*B+C=137, limites e custo. O ótimo é único. Candidatos viáveis são finitos pois B<=19.

### T3 — Versões de pedidos antes do filtro

Manipulação e Filtragem de Dados · hard · `json_schema`

**Prompt exato:**

```text
Considere este CSV. valor_centavos e versao são inteiros. Para cada pedido,
retenha primeiro APENAS a linha de maior versao (não há empates). Só depois
filtre regiao=SUL, status=PAGO e valor_centavos>=10000. Agrupe por cliente,
somando valor_centavos e contando pedidos retidos. Ordene por total_centavos
decrescente; em empate, cliente em ordem alfabética crescente.
Retorne somente uma lista JSON de objetos com cliente, total_centavos e
quantidade. Não inclua explicações ou Markdown.

pedido,versao,cliente,regiao,status,valor_centavos
P01,1,Ana,SUL,PAGO,18000
P02,1,Beto,SUL,PAGO,12000
P03,1,Ana,SUL,PAGO,15000
P01,2,Ana,SUL,CANCELADO,18000
P04,1,Caio,SUL,PAGO,25000
P02,2,Beto,SUL,PAGO,17000
P05,1,Ana,NORTE,PAGO,40000
P06,1,Beto,SUL,PENDENTE,8000
P07,1,Ana,SUL,PAGO,10000
P08,1,Caio,SUL,PAGO,9999
P09,1,Dora,SUL,PAGO,25000
P06,2,Beto,SUL,PAGO,8000
P03,2,Ana,SUL,PAGO,16000
P10,1,Beto,SUL,PAGO,9000
```

**Gabarito determinístico:**

```json
[
  {
    "cliente": "Ana",
    "total_centavos": 26000,
    "quantidade": 2
  },
  {
    "cliente": "Caio",
    "total_centavos": 25000,
    "quantidade": 1
  },
  {
    "cliente": "Dora",
    "total_centavos": 25000,
    "quantidade": 1
  },
  {
    "cliente": "Beto",
    "total_centavos": 17000,
    "quantidade": 1
  }
]
```

**Critério:** Lista de quatro objetos com exatamente os campos indicados. Tipos e valores devem coincidir com o gabarito; ordem da lista obrigatória, ordem das chaves irrelevante. Descartar versões antigas antes de aplicar filtros.

### T4 — Estoque disponível em depósitos habilitados

Manipulação e Filtragem de Dados · medium · `json_schema`

**Prompt exato:**

```text
Use apenas os dados JSON abaixo. Para cada produto ativo de category="sensor",
    some on_hand-reserved somente em depósitos cujo enabled seja true. Todos os
    depósitos referenciados existem e reserved nunca excede on_hand. Retenha
    somente produtos com disponibilidade total >=6. Ordene por disponibilidade
    decrescente e, em empate, sku crescente. Retorne somente uma lista JSON de
    objetos com sku e available (inteiro), sem campos extras ou Markdown.
    Dados:
{"warehouses": [{"id": "W1", "enabled": true}, {"id": "W2", "enabled": false}, {"id": "W3", "enabled": true}], "products": [{"sku": "A1", "active": true, "category": "sensor", "stock": [{"warehouse": "W1", "on_hand": 9, "reserved": 4}, {"warehouse": "W2", "on_hand": 100, "reserved": 0}, {"warehouse": "W3", "on_hand": 5, "reserved": 3}]}, {"sku": "B2", "active": true, "category": "sensor", "stock": [{"warehouse": "W1", "on_hand": 3, "reserved": 1}, {"warehouse": "W3", "on_hand": 7, "reserved": 2}]}, {"sku": "C3", "active": false, "category": "sensor", "stock": [{"warehouse": "W1", "on_hand": 30, "reserved": 0}]}, {"sku": "D4", "active": true, "category": "cable", "stock": [{"warehouse": "W1", "on_hand": 30, "reserved": 0}]}, {"sku": "E5", "active": true, "category": "sensor", "stock": [{"warehouse": "W1", "on_hand": 10, "reserved": 1}, {"warehouse": "W3", "on_hand": 2, "reserved": 2}]}, {"sku": "F6", "active": true, "category": "sensor", "stock": [{"warehouse": "W1", "on_hand": 5, "reserved": 5}, {"warehouse": "W3", "on_hand": 6, "reserved": 1}]}]}
```

**Gabarito determinístico:**

```json
[
  {
    "sku": "E5",
    "available": 9
  },
  {
    "sku": "A1",
    "available": 7
  },
  {
    "sku": "B2",
    "available": 7
  }
]
```

**Critério:** Exigir lista ordenada exatamente como o gabarito; cada objeto possui apenas sku (string) e available (inteiro). Produtos inativos, categorias distintas e estoque de W2 devem ser excluídos.

### T5 — Devolução parcial por elegibilidade

Intenção e Roteamento de Múltiplas Ferramentas · medium · `json_schema`

**Problema comum:**

```text
O usuário quer devolver as linhas L1 e L2 do pedido O17.
Os motivos informados são damaged para L1 e changed_mind para L2.
Política obrigatória: consulte o pedido uma vez. Se estiver entregue, não cancele.
Para cada linha solicitada, consulte check_return em ordem crescente de line_id.
Não reavalie a elegibilidade por conta própria. Depois de todas as consultas,
crie uma única devolução com as linhas cujo eligible=true, em ordem crescente.
Não chame refund_payment: a devolução ainda precisa de inspeção.
```

**Prompt RAW completo:**

```text
O usuário quer devolver as linhas L1 e L2 do pedido O17.
Os motivos informados são damaged para L1 e changed_mind para L2.
Política obrigatória: consulte o pedido uma vez. Se estiver entregue, não cancele.
Para cada linha solicitada, consulte check_return em ordem crescente de line_id.
Não reavalie a elegibilidade por conta própria. Depois de todas as consultas,
crie uma única devolução com as linhas cujo eligible=true, em ordem crescente.
Não chame refund_payment: a devolução ainda precisa de inspeção.

Catálogo (nomes e argumentos exatos; campos extras proibidos):
lookup_order(order_id: string) — Consultar estado e linhas de um pedido.
check_return(line_id: string, reason: string) — Consultar elegibilidade de uma linha e motivo; retorna eligible booleano.
create_return(order_id: string, line_ids: lista não vazia de strings únicas) — Criar uma devolução para as linhas elegíveis; retorna ok booleano.
cancel_order(order_id: string) — Solicitar cancelamento de um pedido.
refund_payment(order_id: string) — Solicitar estorno de um pagamento.

Condição RAW: não execute ferramentas reais. Use os retornos simulados abaixo
para produzir um plano completo, incluindo somente as chamadas necessárias.
Cada retorno está associado à chamada exata em que seria obtido:
[
  {
    "tool": "lookup_order",
    "arguments": {
      "order_id": "O17"
    },
    "result": {
      "status": "delivered",
      "lines": [
        "L1",
        "L2"
      ]
    }
  },
  {
    "tool": "check_return",
    "arguments": {
      "line_id": "L1",
      "reason": "damaged"
    },
    "result": {
      "eligible": true
    }
  },
  {
    "tool": "check_return",
    "arguments": {
      "line_id": "L2",
      "reason": "changed_mind"
    },
    "result": {
      "eligible": false
    }
  },
  {
    "tool": "create_return",
    "arguments": {
      "order_id": "O17",
      "line_ids": [
        "L1"
      ]
    },
    "result": {
      "ok": true
    }
  }
]

Retorne somente uma lista JSON na ordem de execução. Cada elemento deve ter
exatamente tool (nome) e arguments (objeto de argumentos). Sem Markdown ou explicações.
```

**Prompt TOOLS completo:**

```text
O usuário quer devolver as linhas L1 e L2 do pedido O17.
Os motivos informados são damaged para L1 e changed_mind para L2.
Política obrigatória: consulte o pedido uma vez. Se estiver entregue, não cancele.
Para cada linha solicitada, consulte check_return em ordem crescente de line_id.
Não reavalie a elegibilidade por conta própria. Depois de todas as consultas,
crie uma única devolução com as linhas cujo eligible=true, em ordem crescente.
Não chame refund_payment: a devolução ainda precisa de inspeção.

Catálogo (nomes e argumentos exatos; campos extras proibidos):
lookup_order(order_id: string) — Consultar estado e linhas de um pedido.
check_return(line_id: string, reason: string) — Consultar elegibilidade de uma linha e motivo; retorna eligible booleano.
create_return(order_id: string, line_ids: lista não vazia de strings únicas) — Criar uma devolução para as linhas elegíveis; retorna ok booleano.
cancel_order(order_id: string) — Solicitar cancelamento de um pedido.
refund_payment(order_id: string) — Solicitar estorno de um pagamento.

Condição TOOLS: resolva o problema usando as ferramentas locais disponibilizadas.
Faça uma chamada por vez com os argumentos exigidos, aguarde o retorno real
do ambiente e só então decida o próximo passo conforme as políticas.
Não simule retornos, não substitua chamadas por um plano textual e não faça
chamadas paralelas. Use somente chamadas necessárias.

Quando o fluxo terminar, retorne somente um objeto JSON com exatamente:
order_id (string), line_ids (lista das linhas incluídas na devolução criada,
em ordem crescente, ou lista vazia se nenhuma foi criada), status
("return_created" se create_return confirmou ok=true; senão "not_created").
Baseie a resposta nos resultados recebidos; não declare sucesso antes da confirmação.
```

**Catálogo público de ferramentas:**

```json
[
  {
    "name": "lookup_order",
    "description": "Consultar estado e linhas de um pedido.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "order_id"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "check_return",
    "description": "Consultar elegibilidade de uma linha e motivo; retorna eligible booleano.",
    "parameters": {
      "type": "object",
      "properties": {
        "line_id": {
          "type": "string",
          "minLength": 1
        },
        "reason": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "line_id",
        "reason"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "create_return",
    "description": "Criar uma devolução para as linhas elegíveis; retorna ok booleano.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "minLength": 1
        },
        "line_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "minItems": 1,
          "uniqueItems": true
        }
      },
      "required": [
        "order_id",
        "line_ids"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "cancel_order",
    "description": "Solicitar cancelamento de um pedido.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "order_id"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "refund_payment",
    "description": "Solicitar estorno de um pagamento.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "order_id"
      ],
      "additionalProperties": false
    }
  }
]
```

**Fixtures privados do ambiente — nunca enviar como contexto TOOLS:**

```json
{
  "cases": [
    {
      "id": "C1",
      "tool": "lookup_order",
      "arguments": {
        "order_id": "O17"
      },
      "requires": [],
      "result": {
        "status": "delivered",
        "lines": [
          "L1",
          "L2"
        ]
      }
    },
    {
      "id": "C2",
      "tool": "check_return",
      "arguments": {
        "line_id": "L1",
        "reason": "damaged"
      },
      "requires": [
        "C1"
      ],
      "result": {
        "eligible": true
      }
    },
    {
      "id": "C3",
      "tool": "check_return",
      "arguments": {
        "line_id": "L2",
        "reason": "changed_mind"
      },
      "requires": [
        "C1",
        "C2"
      ],
      "result": {
        "eligible": false
      }
    },
    {
      "id": "C4",
      "tool": "create_return",
      "arguments": {
        "order_id": "O17",
        "line_ids": [
          "L1"
        ]
      },
      "requires": [
        "C1",
        "C2",
        "C3"
      ],
      "result": {
        "ok": true
      }
    }
  ],
  "blocked_tools": [
    "cancel_order",
    "refund_payment"
  ],
  "errors": {
    "unknown_tool": {
      "ok": false,
      "error": "unknown_tool"
    },
    "invalid_arguments": {
      "ok": false,
      "error": "invalid_arguments"
    },
    "policy_violation": {
      "ok": false,
      "error": "policy_violation"
    },
    "duplicate_call": {
      "ok": false,
      "error": "duplicate_call"
    },
    "precondition_failed": {
      "ok": false,
      "error": "precondition_failed"
    }
  },
  "dispatch_rules": [
    "Reinicializar o histórico e os fixtures para cada execução de tarefa; não compartilhar estado entre condições ou modelos.",
    "Nome ausente de tool_catalog: retornar errors.unknown_tool.",
    "Validar arguments contra parameters do catálogo, sem coerção e sem tratar booleanos como inteiros; se inválido, retornar errors.invalid_arguments.",
    "Se a ferramenta está em blocked_tools, retornar errors.policy_violation; a regra cobre qualquer argumento válido pelo esquema.",
    "Procurar exatamente um caso por igualdade estrutural de tool e arguments, ignorando só a ordem de chaves; sem caso, retornar errors.invalid_arguments.",
    "Se o caso já foi consumido, retornar errors.duplicate_call.",
    "Se algum ID de requires ainda não foi consumido, retornar errors.precondition_failed.",
    "Caso contrário, retornar uma cópia de result e marcar o caso como consumido, mesmo quando result.ok=false: uma falha de negócio é um resultado esperado.",
    "Respostas de errors não consomem casos. Registrar todas as chamadas e respostas, inclusive erros e tentativas repetidas, para avaliação."
  ]
}
```

**Plano esperado RAW e sequência esperada TOOLS:**

```json
[
  {
    "tool": "lookup_order",
    "arguments": {
      "order_id": "O17"
    }
  },
  {
    "tool": "check_return",
    "arguments": {
      "line_id": "L1",
      "reason": "damaged"
    }
  },
  {
    "tool": "check_return",
    "arguments": {
      "line_id": "L2",
      "reason": "changed_mind"
    }
  },
  {
    "tool": "create_return",
    "arguments": {
      "order_id": "O17",
      "line_ids": [
        "L1"
      ]
    }
  }
]
```

**Desfecho de negócio esperado; resposta final TOOLS:**

```json
{
  "order_id": "O17",
  "line_ids": [
    "L1"
  ],
  "status": "return_created"
}
```

**Avaliação RAW:** Interpretar a saída como uma lista JSON, sem prosa ou Markdown. Exigir igualdade estrutural exata com expected_answer e expected_tool_sequence: ordem das chamadas e listas relevante, ordem das chaves irrelevante, sem coerção de tipos, booleanos distintos de inteiros, sem campos extras ou chamadas adicionais. RAW não executa chamadas nem deve emitir o objeto final de TOOLS. O plano correto aplicado aos fixtures implica final_expected_answer; esse é o mesmo desfecho de negócio avaliado em TOOLS.

**Avaliação TOOLS:** Avaliar o trace capturado pelo ambiente, nunca uma lista de chamadas apenas impressa pelo modelo. Registrar separadamente: (1) seleção de ferramentas comparada aos nomes de expected_tool_sequence; (2) ordem exata das chamadas; (3) validade de cada objeto arguments pelo catálogo e igualdade com os argumentos esperados; (4) número de chamadas extras, inválidas, repetidas ou bloqueadas; (5) uso dos retornos: uma decisão só pode ocorrer após os resultados de requires, com valores recebidos reutilizados corretamente; (6) igualdade estrutural da resposta final com final_expected_answer. Acerto integral exige trace exato com os resultados determinísticos, zero chamadas extras e resposta final correta, sem Markdown ou prosa. Contar todas as chamadas, inclusive as que recebem errors. O trace contém o ramo escolhido com base nos resultados, não prova o raciocínio interno. Excluir a linha não elegível e confirmar a criação.

### T6 — Compensação após falha de pagamento

Intenção e Roteamento de Múltiplas Ferramentas · hard · `json_schema`

**Problema comum:**

```text
Processe a compra de 3 unidades de SKU7 para o pedido O42 por 4200 centavos.
Regras: primeiro consulte estoque; só reserve se available>=quantity.
Só cobre depois da reserva bem-sucedida. Se a cobrança falhar, libere a
reserva e depois registre a falha usando o error retornado como reason.
Não repita a cobrança. Em caso de falha, não emita nota nem recibo.
```

**Prompt RAW completo:**

```text
Processe a compra de 3 unidades de SKU7 para o pedido O42 por 4200 centavos.
Regras: primeiro consulte estoque; só reserve se available>=quantity.
Só cobre depois da reserva bem-sucedida. Se a cobrança falhar, libere a
reserva e depois registre a falha usando o error retornado como reason.
Não repita a cobrança. Em caso de falha, não emita nota nem recibo.

Catálogo (nomes e argumentos exatos; campos extras proibidos):
get_stock(sku: string) — Consultar unidades disponíveis de um SKU; retorna available inteiro.
reserve_stock(order_id: string, sku: string, quantity: inteiro positivo) — Reservar unidades para um pedido; retorna ok e reservation_id quando bem-sucedida.
charge(order_id: string, amount_centavos: inteiro positivo) — Cobrar um pedido em centavos; retorna ok e error em caso de falha.
release_stock(reservation_id: string) — Liberar uma reserva pelo identificador recebido; retorna ok.
log_failure(order_id: string, reason: string) — Registrar uma falha com o motivo recebido; retorna ok.
issue_invoice(order_id: string) — Emitir uma nota para um pedido.
send_receipt(order_id: string) — Enviar um recibo para um pedido.

Condição RAW: não execute ferramentas reais. Use os retornos simulados abaixo
para produzir um plano completo, incluindo somente as chamadas necessárias.
Cada retorno está associado à chamada exata em que seria obtido:
[
  {
    "tool": "get_stock",
    "arguments": {
      "sku": "SKU7"
    },
    "result": {
      "available": 5
    }
  },
  {
    "tool": "reserve_stock",
    "arguments": {
      "order_id": "O42",
      "sku": "SKU7",
      "quantity": 3
    },
    "result": {
      "ok": true,
      "reservation_id": "R9"
    }
  },
  {
    "tool": "charge",
    "arguments": {
      "order_id": "O42",
      "amount_centavos": 4200
    },
    "result": {
      "ok": false,
      "error": "payment_declined"
    }
  },
  {
    "tool": "release_stock",
    "arguments": {
      "reservation_id": "R9"
    },
    "result": {
      "ok": true
    }
  },
  {
    "tool": "log_failure",
    "arguments": {
      "order_id": "O42",
      "reason": "payment_declined"
    },
    "result": {
      "ok": true
    }
  }
]

Retorne somente uma lista JSON na ordem de execução. Cada elemento deve ter
exatamente tool (nome) e arguments (objeto de argumentos). Sem Markdown ou explicações.
```

**Prompt TOOLS completo:**

```text
Processe a compra de 3 unidades de SKU7 para o pedido O42 por 4200 centavos.
Regras: primeiro consulte estoque; só reserve se available>=quantity.
Só cobre depois da reserva bem-sucedida. Se a cobrança falhar, libere a
reserva e depois registre a falha usando o error retornado como reason.
Não repita a cobrança. Em caso de falha, não emita nota nem recibo.

Catálogo (nomes e argumentos exatos; campos extras proibidos):
get_stock(sku: string) — Consultar unidades disponíveis de um SKU; retorna available inteiro.
reserve_stock(order_id: string, sku: string, quantity: inteiro positivo) — Reservar unidades para um pedido; retorna ok e reservation_id quando bem-sucedida.
charge(order_id: string, amount_centavos: inteiro positivo) — Cobrar um pedido em centavos; retorna ok e error em caso de falha.
release_stock(reservation_id: string) — Liberar uma reserva pelo identificador recebido; retorna ok.
log_failure(order_id: string, reason: string) — Registrar uma falha com o motivo recebido; retorna ok.
issue_invoice(order_id: string) — Emitir uma nota para um pedido.
send_receipt(order_id: string) — Enviar um recibo para um pedido.

Condição TOOLS: resolva o problema usando as ferramentas locais disponibilizadas.
Faça uma chamada por vez com os argumentos exigidos, aguarde o retorno real
do ambiente e só então decida o próximo passo conforme as políticas.
Não simule retornos, não substitua chamadas por um plano textual e não faça
chamadas paralelas. Use somente chamadas necessárias.

Quando o fluxo terminar, retorne somente um objeto JSON com exatamente:
order_id (string), status ("payment_failed" se a cobrança falhou,
"completed" se o fluxo concluiu com sucesso, ou "insufficient_stock" se faltou estoque),
error (string recebida da cobrança em caso de falha, senão null),
reservation_released (booleano: liberação confirmada por ok=true),
failure_logged (booleano: registro de falha confirmado por ok=true).
Baseie esses campos somente nas chamadas e resultados efetivamente recebidos.
```

**Catálogo público de ferramentas:**

```json
[
  {
    "name": "get_stock",
    "description": "Consultar unidades disponíveis de um SKU; retorna available inteiro.",
    "parameters": {
      "type": "object",
      "properties": {
        "sku": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "sku"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "reserve_stock",
    "description": "Reservar unidades para um pedido; retorna ok e reservation_id quando bem-sucedida.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "minLength": 1
        },
        "sku": {
          "type": "string",
          "minLength": 1
        },
        "quantity": {
          "type": "integer",
          "minimum": 1
        }
      },
      "required": [
        "order_id",
        "sku",
        "quantity"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "charge",
    "description": "Cobrar um pedido em centavos; retorna ok e error em caso de falha.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "minLength": 1
        },
        "amount_centavos": {
          "type": "integer",
          "minimum": 1
        }
      },
      "required": [
        "order_id",
        "amount_centavos"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "release_stock",
    "description": "Liberar uma reserva pelo identificador recebido; retorna ok.",
    "parameters": {
      "type": "object",
      "properties": {
        "reservation_id": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "reservation_id"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "log_failure",
    "description": "Registrar uma falha com o motivo recebido; retorna ok.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "minLength": 1
        },
        "reason": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "order_id",
        "reason"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "issue_invoice",
    "description": "Emitir uma nota para um pedido.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "order_id"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "send_receipt",
    "description": "Enviar um recibo para um pedido.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "order_id"
      ],
      "additionalProperties": false
    }
  }
]
```

**Fixtures privados do ambiente — nunca enviar como contexto TOOLS:**

```json
{
  "cases": [
    {
      "id": "C1",
      "tool": "get_stock",
      "arguments": {
        "sku": "SKU7"
      },
      "requires": [],
      "result": {
        "available": 5
      }
    },
    {
      "id": "C2",
      "tool": "reserve_stock",
      "arguments": {
        "order_id": "O42",
        "sku": "SKU7",
        "quantity": 3
      },
      "requires": [
        "C1"
      ],
      "result": {
        "ok": true,
        "reservation_id": "R9"
      }
    },
    {
      "id": "C3",
      "tool": "charge",
      "arguments": {
        "order_id": "O42",
        "amount_centavos": 4200
      },
      "requires": [
        "C1",
        "C2"
      ],
      "result": {
        "ok": false,
        "error": "payment_declined"
      }
    },
    {
      "id": "C4",
      "tool": "release_stock",
      "arguments": {
        "reservation_id": "R9"
      },
      "requires": [
        "C1",
        "C2",
        "C3"
      ],
      "result": {
        "ok": true
      }
    },
    {
      "id": "C5",
      "tool": "log_failure",
      "arguments": {
        "order_id": "O42",
        "reason": "payment_declined"
      },
      "requires": [
        "C1",
        "C2",
        "C3",
        "C4"
      ],
      "result": {
        "ok": true
      }
    }
  ],
  "blocked_tools": [
    "issue_invoice",
    "send_receipt"
  ],
  "errors": {
    "unknown_tool": {
      "ok": false,
      "error": "unknown_tool"
    },
    "invalid_arguments": {
      "ok": false,
      "error": "invalid_arguments"
    },
    "policy_violation": {
      "ok": false,
      "error": "policy_violation"
    },
    "duplicate_call": {
      "ok": false,
      "error": "duplicate_call"
    },
    "precondition_failed": {
      "ok": false,
      "error": "precondition_failed"
    }
  },
  "dispatch_rules": [
    "Reinicializar o histórico e os fixtures para cada execução de tarefa; não compartilhar estado entre condições ou modelos.",
    "Nome ausente de tool_catalog: retornar errors.unknown_tool.",
    "Validar arguments contra parameters do catálogo, sem coerção e sem tratar booleanos como inteiros; se inválido, retornar errors.invalid_arguments.",
    "Se a ferramenta está em blocked_tools, retornar errors.policy_violation; a regra cobre qualquer argumento válido pelo esquema.",
    "Procurar exatamente um caso por igualdade estrutural de tool e arguments, ignorando só a ordem de chaves; sem caso, retornar errors.invalid_arguments.",
    "Se o caso já foi consumido, retornar errors.duplicate_call.",
    "Se algum ID de requires ainda não foi consumido, retornar errors.precondition_failed.",
    "Caso contrário, retornar uma cópia de result e marcar o caso como consumido, mesmo quando result.ok=false: uma falha de negócio é um resultado esperado.",
    "Respostas de errors não consomem casos. Registrar todas as chamadas e respostas, inclusive erros e tentativas repetidas, para avaliação."
  ]
}
```

**Plano esperado RAW e sequência esperada TOOLS:**

```json
[
  {
    "tool": "get_stock",
    "arguments": {
      "sku": "SKU7"
    }
  },
  {
    "tool": "reserve_stock",
    "arguments": {
      "order_id": "O42",
      "sku": "SKU7",
      "quantity": 3
    }
  },
  {
    "tool": "charge",
    "arguments": {
      "order_id": "O42",
      "amount_centavos": 4200
    }
  },
  {
    "tool": "release_stock",
    "arguments": {
      "reservation_id": "R9"
    }
  },
  {
    "tool": "log_failure",
    "arguments": {
      "order_id": "O42",
      "reason": "payment_declined"
    }
  }
]
```

**Desfecho de negócio esperado; resposta final TOOLS:**

```json
{
  "order_id": "O42",
  "status": "payment_failed",
  "error": "payment_declined",
  "reservation_released": true,
  "failure_logged": true
}
```

**Avaliação RAW:** Interpretar a saída como uma lista JSON, sem prosa ou Markdown. Exigir igualdade estrutural exata com expected_answer e expected_tool_sequence: ordem das chamadas e listas relevante, ordem das chaves irrelevante, sem coerção de tipos, booleanos distintos de inteiros, sem campos extras ou chamadas adicionais. RAW não executa chamadas nem deve emitir o objeto final de TOOLS. O plano correto aplicado aos fixtures implica final_expected_answer; esse é o mesmo desfecho de negócio avaliado em TOOLS.

**Avaliação TOOLS:** Avaliar o trace capturado pelo ambiente, nunca uma lista de chamadas apenas impressa pelo modelo. Registrar separadamente: (1) seleção de ferramentas comparada aos nomes de expected_tool_sequence; (2) ordem exata das chamadas; (3) validade de cada objeto arguments pelo catálogo e igualdade com os argumentos esperados; (4) número de chamadas extras, inválidas, repetidas ou bloqueadas; (5) uso dos retornos: uma decisão só pode ocorrer após os resultados de requires, com valores recebidos reutilizados corretamente; (6) igualdade estrutural da resposta final com final_expected_answer. Acerto integral exige trace exato com os resultados determinísticos, zero chamadas extras e resposta final correta, sem Markdown ou prosa. Contar todas as chamadas, inclusive as que recebem errors. O trace contém o ramo escolhido com base nos resultados, não prova o raciocínio interno. Usar o reservation_id recebido na liberação e o error recebido no registro, aguardando confirmação de ambas as operações.

### T7 — Reconciliação sem mutar eventos

Programação e Automação · hard · `code`

**Prompt exato:**

```text
Escreva em Python 3.9+, usando apenas a biblioteca padrão, a função
reconcile(events). events é uma lista de dicionários com id (string), seq
(inteiro não negativo), status ("settled" ou "void") e amount_centavos
(inteiro, inclusive negativo ou zero). Todos os dados são válidos; para
cada id, seq não se repete. A lista pode estar fora de ordem ou vazia.
Para cada id, retenha primeiro o evento de maior seq. Depois selecione
apenas os eventos retidos com status="settled". Retorne um dicionário
com net_centavos (soma de amount_centavos desses eventos) e ids (lista
dos respectivos ids ordenada lexicograficamente). Um evento settled com
valor zero deve aparecer em ids. Para entrada vazia, retorne soma zero
e ids vazio. Não altere a lista recebida nem seus dicionários.
Não leia arquivos, rede ou stdin e não imprima. Retorne somente código
Python, sem Markdown, que defina a função; imports padrão são permitidos.
```

**Gabarito determinístico:**

```json
{
  "function": "reconcile",
  "reference_code": "def reconcile(events):\n    latest = {}\n    for event in events:\n        if event[\"id\"] not in latest or event[\"seq\"] > latest[event[\"id\"]][\"seq\"]:\n            latest[event[\"id\"]] = event\n    settled = [event for event in latest.values() if event[\"status\"] == \"settled\"]\n    return {\"net_centavos\": sum(event[\"amount_centavos\"] for event in settled),\n            \"ids\": sorted(event[\"id\"] for event in settled)}\n",
  "test_cases": [
    {
      "input": [],
      "expected": {
        "net_centavos": 0,
        "ids": []
      }
    },
    {
      "input": [
        {
          "id": "B",
          "seq": 2,
          "status": "void",
          "amount_centavos": 900
        },
        {
          "id": "A",
          "seq": 1,
          "status": "settled",
          "amount_centavos": 1000
        },
        {
          "id": "B",
          "seq": 1,
          "status": "settled",
          "amount_centavos": 900
        },
        {
          "id": "C",
          "seq": 3,
          "status": "settled",
          "amount_centavos": -250
        },
        {
          "id": "A",
          "seq": 2,
          "status": "settled",
          "amount_centavos": 1200
        }
      ],
      "expected": {
        "net_centavos": 950,
        "ids": [
          "A",
          "C"
        ]
      }
    },
    {
      "input": [
        {
          "id": "Z",
          "seq": 8,
          "status": "settled",
          "amount_centavos": 0
        },
        {
          "id": "Z",
          "seq": 2,
          "status": "void",
          "amount_centavos": 40
        }
      ],
      "expected": {
        "net_centavos": 0,
        "ids": [
          "Z"
        ]
      }
    },
    {
      "input": [
        {
          "id": "X",
          "seq": 1,
          "status": "settled",
          "amount_centavos": 12
        },
        {
          "id": "X",
          "seq": 7,
          "status": "void",
          "amount_centavos": 12
        }
      ],
      "expected": {
        "net_centavos": 0,
        "ids": []
      }
    }
  ]
}
```

**Critério:** Avaliar comportamento da função, não igualdade textual ao código de referência. Todos os test_cases devem passar com igualdade de tipos e valores; usar cópia profunda antes da chamada e verificar entrada inalterada. Verificar também entradas válidas adicionais dentro do domínio do prompt; não exigir tratamento de entradas inválidas. Execução posterior deve ser isolada; nenhum runner é criado nesta versão.

### T8 — Expansão ordenada de intervalos

Programação e Automação · hard · `code`

**Prompt exato:**

```text
Implemente expand_ranges(text) em Python 3.9+, só com biblioteca padrão.
text é uma string de tokens separados por vírgula. Remova espaços em branco
das extremidades de cada token. Um token válido é um inteiro decimal ASCII
com sinal "-" opcional, ou dois desses inteiros separados exatamente por
"..", sem espaços internos. O sinal "+" não é permitido. Zeros iniciais
são permitidos. Intervalos incluem os dois extremos e podem ser crescentes
ou decrescentes. Expanda na ordem dos tokens; remova repetições preservando
somente a primeira aparição de cada inteiro. Exemplo:
"3..1, 2, -2..0, 3" -> [3, 2, 1, -2, -1, 0].
Entrada vazia ou contendo só espaços retorna []. Qualquer token inválido,
inclusive token vazio entre vírgulas, causa ValueError (mensagem livre).
Em entradas válidas há no máximo 1000 números antes de remover repetições.
Retorne somente código que defina a função, sem Markdown, leitura de
arquivos/rede/stdin ou prints. Imports padrão são permitidos.
```

**Gabarito determinístico:**

```json
{
  "function": "expand_ranges",
  "reference_code": "import re\n\ndef expand_ranges(text):\n    if not text.strip():\n        return []\n    result, seen = [], set()\n    for raw_token in text.split(\",\"):\n        token = raw_token.strip()\n        match = re.fullmatch(r\"(-?[0-9]+)(?:\\.\\.(-?[0-9]+))?\", token)\n        if not match:\n            raise ValueError(\"invalid token\")\n        start = int(match.group(1))\n        end = start if match.group(2) is None else int(match.group(2))\n        step = 1 if end >= start else -1\n        for value in range(start, end + step, step):\n            if value not in seen:\n                result.append(value)\n                seen.add(value)\n    return result\n",
  "test_cases": [
    {
      "input": "3..1, 2, -2..0, 3",
      "expected": [
        3,
        2,
        1,
        -2,
        -1,
        0
      ]
    },
    {
      "input": "  ",
      "expected": []
    },
    {
      "input": "-1..-3,-2,0..0",
      "expected": [
        -1,
        -2,
        -3,
        0
      ]
    },
    {
      "input": "2,2,1..3",
      "expected": [
        2,
        1,
        3
      ]
    },
    {
      "input": "1,,2",
      "raises": "ValueError"
    },
    {
      "input": "1..2..3",
      "raises": "ValueError"
    },
    {
      "input": "1 ..2",
      "raises": "ValueError"
    },
    {
      "input": "+2",
      "raises": "ValueError"
    }
  ]
}
```

**Critério:** Executar test_cases e comparar listas de inteiros na ordem exata; casos raises devem lançar ValueError. Mensagem da exceção e texto do código são livres. Validar também extremos iguais, zeros iniciais e vírgula final conforme a especificação. Não usar modelos ou rede; a futura execução de código gerado deve ser isolada.

### T9 — Ata consolidada com revisões

Síntese e Formatação Estruturada Restrita · hard · `json_schema`

**Prompt exato:**

```text
Converta as notas abaixo em um único objeto JSON. Use somente os campos
decision_ids, actions e blockers; não escreva texto fora do JSON.
Vocabulário fechado:
decisões D1=colecao_unica, D2=planilha_externa;
tarefas A1=preparar_dados, A2=revisar_dados, A3=publicar_piloto;
bloqueio B1=token_ausente; pessoas Ana, Bruno, Carla.
Regras: inclua apenas decisões confirmadas e ações não canceladas. A última
revisão explícita de responsável ou prazo substitui a anterior. Ordene IDs
em ordem crescente. actions é uma lista de objetos com exatamente id,
owner, due_day (inteiro relativo ao dia zero) e depends_on (lista de IDs).
blockers é uma lista de objetos com exatamente id, owner, blocks (lista
de IDs). Listas internas também devem estar em ordem crescente.
Notas, em ordem cronológica:
- D1 foi confirmada. D2 foi proposta, mas rejeitada.
- A1 foi atribuída a Ana para o dia 3, sem dependências.
- A2 foi atribuída a Bruno para o dia 5, dependendo somente de A1.
- A3 foi atribuída a Carla para o dia 6, dependendo somente de A2.
- O prazo de A1 mudou para o dia 4; Ana continua responsável.
- B1 está aberto, Carla é responsável; ele bloqueia somente A3.
- A2 foi transferida para Carla, mantendo prazo e dependências.
- A3 foi cancelada. B1 continua aberto, mas referências a ações canceladas
  devem ser removidas de blocks; não exclua um bloqueio aberto por ficar vazio.
```

**Gabarito determinístico:**

```json
{
  "decision_ids": [
    "D1"
  ],
  "actions": [
    {
      "id": "A1",
      "owner": "Ana",
      "due_day": 4,
      "depends_on": []
    },
    {
      "id": "A2",
      "owner": "Carla",
      "due_day": 5,
      "depends_on": [
        "A1"
      ]
    }
  ],
  "blockers": [
    {
      "id": "B1",
      "owner": "Carla",
      "blocks": []
    }
  ]
}
```

**Critério:** Objeto JSON estrito com os três campos indicados e sem campos extras em qualquer nível. Exigir todos os tipos, valores e ordens de listas do gabarito. Não aceitar a decisão rejeitada, a ação cancelada, responsáveis ou prazos anteriores.

### T10 — Boletim operacional em oito linhas

Síntese e Formatação Estruturada Restrita · hard · `structured_text`

**Prompt exato:**

```text
Gere um boletim com exatamente oito linhas, sem Markdown, linhas vazias ou
espaços no início/fim. Use apenas os registros e regras abaixo.
Há quatro serviços: auth, billing, search, storage. Cada registro substitui
integralmente o estado anterior do mesmo serviço; use o maior número de
revisão. Severidades possíveis: critical, high, low, none. Apenas status=open
é incidente ativo. Para incidentes ativos, prioridade: critical antes de
high antes de low; em empate, serviço em ordem alfabética. Não inclua
resolved nem false_alarm na lista de ativos.
Registros (serviço; revisão; status; severidade):
auth;1;open;critical
billing;1;open;low
storage;1;open;high
search;1;open;high
auth;2;resolved;none
billing;3;open;high
search;2;false_alarm;none
storage;2;open;high
billing;2;open;critical
Formato exato de cada linha (substitua os valores entre <>):
STATUS=<ALERT se houver algum ativo, senão OK>
ACTIVE=<quantidade inteira de incidentes ativos>
CRITICAL=<quantidade inteira de ativos critical>
HIGH=<quantidade inteira de ativos high>
LOW=<quantidade inteira de ativos low>
SERVICES=<serviços ativos na ordem de prioridade, separados por vírgula sem espaços; ou none>
FIRST=<primeiro serviço de SERVICES; ou none>
ACTION=<ESCALATE se houver critical ou high ativo; senão MONITOR se houver low ativo; senão NONE>
```

**Gabarito determinístico:**

```json
"STATUS=ALERT\nACTIVE=2\nCRITICAL=0\nHIGH=2\nLOW=0\nSERVICES=billing,storage\nFIRST=billing\nACTION=ESCALATE"
```

**Critério:** Normalizar somente CRLF para LF e aceitar um único LF final opcional. Após isso, exigir igualdade textual com o gabarito (oito linhas). Não remover outros espaços nem aceitar Markdown, chaves extras ou severidade antiga de billing.
