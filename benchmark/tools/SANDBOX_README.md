# Backend seguro de código da Fase 2

As APIs são run_python_executor(arguments, backend=None),
run_python_test_runner(arguments, backend=None) e run_python_syntax_check(arguments).
As duas primeiras exigem SandboxBackend; DockerSandboxBackend é a implementação
disponível. validate_phase2_environment() verifica disponibilidade, isolamento,
limites cgroup v2, runtime e timeout/cancelamento, antes de um futuro runner
carregar modelos. Ausência de Docker, da imagem local ou das capacidades exigidas
é erro de infraestrutura. Não há fallback para Python local.

A imagem padrão está fixada pelo digest:
python@sha256:b658b05ce9c4eb97ec6fc19424818a1f31993659c2b505a532f470e83807815d
(Python 3.12 Alpine). A infraestrutura deve provisioná-la antecipadamente
(docker pull com esse digest); cada chamada usa --pull=never e não instala pacotes.
Um backend alternativo seguro poderá implementar a mesma interface no Colab;
Docker não é pressuposto como disponível lá.

Cada execução cria e remove um container, sem mounts, sem acesso ao socket Docker,
sem rede externa, sem ambiente herdado do host, com root read-only, usuário
65534:65534, capabilities removidas e no-new-privileges. /tmp é tmpfs descartável
de 8 MiB; memória e swap são limitadas a 256 MiB, PIDs a 16 e CPU a um núcleo.
O worker verifica também os recursos efetivos, rede e raiz read-only antes de executar código. A checagem atual de recursos exige cgroup v2; outro ambiente deve implementar
e verificar seu próprio backend antes de executar oficialmente.

Há prazo externo de 2 segundos, incluindo preparação/verificações e startup.
Uma expiração mata o cliente Docker e o container, verifica sua parada e remove-o.
A limpeza verificada pode acrescentar latência ao prazo de execução; não é permitido deixá-la pendente. Falha de cancelamento/cleanup é erro de infraestrutura. cancel()/cleanup() também
permitem cancelamento explícito. Containers têm nomes aleatórios e label
llm-benchmark-lab.sandbox=phase2; nenhum processo de código do modelo roda no host.

A política AST é aplicada antes da criação do container e novamente no worker.
Requer uma função pública alvo (main(data) no executor), permite imports de re,
math, decimal, collections, itertools, functools, operator, statistics, bisect,
heapq, string e json, com membros públicos enumerados em _sandbox_worker.py.
Não expõe módulos reais completos: proxies evitam imports indiretos por atributos
de módulos. Imports relativos/star, reflexão, atributos privados, classes,
decorators e builtins de I/O/execução dinâmica são proibidos. Helpers locais dentro
da função são possíveis. Essa política conservadora pode rejeitar código Python
válido; isso é sandbox_policy, não prova de incapacidade da tarefa.

Somente o worker isolado executa o código submetido. O worker captura toda saída,
não oferece stdin e usa audit hook como defesa adicional contra arquivos,
rede e processos. A allowlist de AST não é considerada fronteira suficiente:
Docker continua obrigatório mesmo quando o código passa pela política.
Rede none isola a rede externa; a namespace ainda pode conter loopback, cujo uso
pelo modelo é bloqueado pela política e pelos imports. /tmp gravável e recursos
do SO são verificados por probes confiáveis, nunca concedidos ao código do modelo.

Testes de integração reais são as classes DockerIntegrationTests e CodeToolsTests.
Não fazem skip silencioso se Docker faltar. Os probes privados de infraestrutura
executam scripts fixos confiáveis para verificar rede, filesystem, cgroups e PIDs;
não são APIs públicas de ferramentas nem recebem código do modelo.

SandboxError mantém code, origin e scope para o controlador:
submitted_code, sandbox_policy ou backend_infrastructure. Mensagens públicas
são fixas, sem traceback, paths, stderr bruto ou mensagens de exceção submetidas.
OOM/limite ou timeout cancelado corretamente são erros do código submetido.
Falha do daemon, imagem, transporte ou cleanup é infraestrutura.
No test runner, exceções por caso aparecem apenas por nome e falham/passam conforme
raises; nunca incluem suas mensagens. Mutação de argumentos ou stdout/stderr
impede aprovação. Cada caso usa cópias profundas; o processo é único para o lote
de até 100 casos e o prazo de 2 s vale para o lote inteiro.

O canal interno tipado preserva bool/int, Decimal e objetos JSON sem colisões
com chaves do usuário. Resultados numéricos Decimal permanecem Decimal no Python,
não viram strings ou float. A contagem do limite de 8192 bytes usa JSON numérico
finito com Decimal como token numérico. Nenhum caso oficial, fixture, prompt ou
arquivo do benchmark é enviado ao worker.

Docker depende do kernel e do daemon; não equivale a uma VM contra código hostil.
O daemon é confiável e nunca é exposto no container. A camada restringe código
Python e reduz recursos, mas não promete tempo de CPU/hardware idêntico entre
ambientes. Startup está incluído no prazo e pode falhar em hosts lentos.
