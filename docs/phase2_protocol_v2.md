# Phase 2 Protocol V2 — revisão anterior à coleta

Esta é a versão selecionada pelo entrypoint `run_phase2.py`. A V1 permanece em
`phase2_protocol.md`, `phase2_tool_catalog.md`, `phase2_frozen.json` e no commit
`2da126c`. O preflight realizado no Colab pelo pesquisador, antes da primeira
observação oficial, encontrou Python 3.13 e ausência de Docker. A V2 permite
Python 3.10–3.13 e troca exclusivamente o mecanismo das ferramentas Python e
requisitos de ambiente. Nenhum dado oficial TOOLS foi coletado nesta revisão.
A versão/hashes desta revisão devem ser aprovados e versionados antes da coleta.
Todos os modelos devem usar a mesma V2; não misturar observações V1/V2.

## Invariantes

Benchmark V1, dez tarefas, modelos, RAW/TOOLS, seleção dos textos públicos,
geração FP16/auto, 512 tokens/turno, greedy, temperature/seed=None, dois warm-ups
descartados, execução sequencial, 8 despachos/9 outputs, contexto 16384/entrada
15872 e supervisão externa de 600 s permanecem. O loop é o mesmo; recebe apenas
routing e serialização de catálogo versionados. Task correctness, strict,
contrato T5/T6, eficiência e fórmulas da V1 não são redefinidos.
Fixtures, respostas e matriz offline continuam exclusivamente privados.

## Interpretador restrito, não CPython dinâmico

`sandbox_mode="restricted_ast_interpreter"`. O programa é parseado por `ast.parse`
e percorrido por handlers fechados. Nunca é passado a eval/exec/bytecode/compile
para execução, nem a um subprocesso Python. Não há imports de código submetido,
objeto host, acesso a atributos arbitrários, rede, filesystem, env, processos,
relógio ou aleatoriedade. O processo supervisionado executa somente o runner,
o modelo e este interpretador confiável; ele não é um fallback para CPython.
Docker, bubblewrap e unshare não participam da execução V2.

O catálogo V2 publica o subset e limites exatos. Suporta funções top-level
posicionais, variáveis, assign/augassign/return, if/for/else/break/continue/pass,
index/slice, expressões limitadas, dict/list/tuple interno e builtins/métodos
explicitamente listados. Literais float e comprehensions não fazem parte do
subset. Decimal finito de dados JSON é preservado; divisão tem precisão 64 e
expoente limitado, sem conversão binária intermediária. Identity só é permitido
com None/bool. Funções sem defaults/annotations/decorators/varargs. Raise somente
das cinco exceções contratuais, com literal de mensagem opcional (não publicado).
Outras construções são recusadas antes de interpretar o corpo.

Limites lógicos independem de wall time: fonte 32768 caracteres; AST 4096 nós,
profundidade 32; 20000 visitas/statements; 10000 iterações; ranges e containers
4096; strings 8192; inteiros 256 bits; 256 chamadas e profundidade 16; 10000
operações aritméticas; 250000 unidades cumulativas de alocação; 20000 unidades
por árvore de dados (incluindo caracteres); saída 8192 bytes. Checagens precedem
potências/concatenação/repetição/conversão; ciclos são recusados. Prazo adicional
2 s. Limites são conservadores e podem recusar programas legítimos, nunca
acionam execução alternativa. Não se afirma equivalência com todo Python.

Os resultados/envelopes/schemas dos tools mantêm a forma congelada. Imports
que antes podiam existir na sandbox Docker agora recebem policy_violation.
Erro de política/limite aplicado corretamente é sucesso técnico da infraestrutura.
Erro do interpretador/controlador é infrastructure_failure; sem fallback.
`python_syntax_check` verifica apenas parsing, sem executar ou aplicar o subset.

## Segurança bloqueante e reprodutibilidade

O preflight exige `restricted_python_runtime=PASS`, executando self-tests positivos
e negativos. Docker ausente não é falha V2; falha do interpretador bloqueia o run.
O manifesto `phase2_frozen_v2.json` valida fontes, catálogo V2, documentos e
compatibilidade. A integridade V1 histórica é auditável pelo commit original;
o preflight V1 não deve autorizar coleta em um checkout com revisão V2.
Não se trata de isolamento OS: o argumento de segurança é o universo fechado de
valores/operações, limites antes de alocar e testes adversariais. Novas operações
exigem revisão, testes e nova versão antes de dados experimentais.

## Correção de T7/T8: contrato original preservado

O Benchmark V1 permite biblioteca padrão. Essa permissão **não é removida** das
respostas finais. Código compatível com o interpretador pode ser verificado com
os mesmos casos congelados da Fase 1. Código fora do subset/limite fica
`evaluation_pending=true`, com conteúdo e acerto desconhecidos (null quando não
há outro erro conclusivo), e requer correção posterior em ambiente isolado.
Esta decisão foi aprovada pelo pesquisador. Um import válido não vira resposta
incorreta apenas por limitação do avaliador. Nenhum programa é executado por
CPython no Colab para resolver essa pendência. Relatórios com pendências têm
`publication_ready=false`; taxas provisórias não são placar final de acurácia.
Conteúdo de casos privados nunca é enviado ao modelo. Integração posterior de
vereditos isolados exige proveniência/revisão, sem reexecutar ou alterar respostas.

## Catálogo

V1 SHA-256: `cfe666d04c4870461bd3c18bfd5b78a393d5778cd6fb2aac6399d6701dade92b`.
V2 SHA-256: `f802970524fa3daadb8e30801aa40ed81995b1b8d0f1fd64053e34b178ddc112`.
Os 25 nomes, argumentos e retornos permanecem. Descrições dos dois executores
explicitam o subset; syntax_check descreve parsing em vez de compilação. A V1
continua serializável byte a byte pelo módulo original; a V2 usa `catalog_v2.py`.
