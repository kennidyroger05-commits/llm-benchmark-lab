"""Provider-independent Phase 2 episodes; no generation, scoring or persistence.

The callback receives one exact JSON document, following protocol section 4.2.
Only public text/catalog and ordered model envelopes/tool results enter it. Raw
outputs and exact callback inputs are retained separately, without repair or
truncation. Private dependencies/metadata never enter the document.

No received output means protocol_correct=None, including an initial callback or
preflight failure. Otherwise it describes all received messages independently of
technical success. A context-budget stop is not an infrastructure failure; a
counter exception is. Without a tokenizer callback, token_count_validated=False
and no character-based estimate is made. There is no real-provider integration
or episode wall-clock cancellation here; tool deadlines remain backend-owned.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Callable

from benchmark.tools.catalog import serialize_public_catalog
from benchmark.tools.domain_tools import DomainSession
from benchmark.tools.sandbox import SandboxBackend
from benchmark.tools._sandbox_worker import json_bytes
from .dispatcher import DispatchContext, dispatch_tool
from .protocol import FinalAnswer, ProtocolError, parse_model_message


# Verbatim operational instruction frozen in phase2_protocol.md, section 4.1.
OPERATIONAL_PROTOCOL = (
    "Cada saída deve ser um objeto JSON completo, sem texto externo. Há dois tipos "
    "de mensagem: tool_call, com exatamente type, tool (string) e arguments "
    "(objeto); final_answer, com exatamente type e answer. Em final_answer, answer "
    "contém a resposta no formato solicitado pela tarefa; código e texto são "
    "strings JSON. A instrução de formato da tarefa aplica-se a answer. Uma "
    "mensagem tool_call solicita uma ferramenta do catálogo; seu retorno será "
    "acrescentado ao histórico antes da próxima saída. O limite é de 8 chamadas "
    "despachadas e 9 saídas do modelo por tarefa. A saída após a oitava chamada "
    "pode ser uma resposta final."
)


@dataclass(frozen=True)
class EpisodeLimits:
    """Official defaults. Smaller call/output budgets permit controlled tests only."""
    max_tool_calls: int = 8
    max_model_outputs: int = 9
    context_tokens: int = 16384
    reserved_output_tokens: int = 512

    def __post_init__(self):
        values = (self.max_tool_calls, self.max_model_outputs,
                  self.context_tokens, self.reserved_output_tokens)
        if (any(type(value) is not int for value in values)
                or not 0 <= self.max_tool_calls <= 8
                or not 1 <= self.max_model_outputs <= 9
                or not 0 < self.reserved_output_tokens < self.context_tokens):
            raise ValueError("Invalid episode limits.")

    @property
    def input_token_limit(self):
        return self.context_tokens - self.reserved_output_tokens


@dataclass
class EpisodeResult:
    """Structural execution evidence, deliberately without task-correctness metrics.

    full_history is the final protocol/catalog/task/history document. Its history
    alternates model objects and public tool envelopes. A terminal invalid raw
    string is preserved there too, but is never sent for another generation.
    model_contexts contains the exact strings passed to the callback; metadata
    and the private trace classification are excluded from those strings.
    """
    technical_success: bool = False
    protocol_correct: bool | None = None
    termination_reason: str = ""
    final_answer: Any = None
    model_outputs_total: int = 0
    tool_calls_dispatched: int = 0
    tool_trace: list[dict] = field(default_factory=list)
    model_messages: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    full_history: dict = field(default_factory=dict)
    direct_answer: bool = False
    infrastructure_failure: bool = False
    error_code: str | None = None
    model_contexts: list[str] = field(default_factory=list)
    context_token_counts: list[int] = field(default_factory=list)
    token_count_validated: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        """Detached structural snapshot; exact Decimal answers retain their type."""
        return asdict(self)


def _public_document(task, public_catalog, catalog_serializer=serialize_public_catalog):
    """Allowlist selection, never a copy/serialization of the task record."""
    if not isinstance(task, dict) or task.get("id") not in {f"T{i}" for i in range(1, 11)}:
        raise ValueError("A task ID from T1 to T10 is required.")
    key = "tools_prompt" if task["id"] in {"T5", "T6"} else "prompt"
    if not isinstance(task.get(key), str) or not task[key]:
        raise ValueError("Required public task prompt is missing.")
    frozen = catalog_serializer()
    supplied = frozen if public_catalog is None else public_catalog
    if isinstance(supplied, str):
        supplied = supplied.encode("utf-8")
    # Reject reordered, edited or extended catalogs, rather than silently fixing
    # them. This guarantees all tasks get exactly the frozen lexical catalog.
    if not isinstance(supplied, bytes) or supplied != frozen:
        raise ValueError("Expected the exact serialized public catalog.")
    return {"protocol": OPERATIONAL_PROTOCOL, "tool_catalog": json.loads(frozen),
            "task": task[key], "history": []}


def run_episode(
    task: dict,
    model_callback: Callable[[str], str],
    *,
    public_catalog: bytes | str | None = None,
    domain_session: DomainSession | None = None,
    sandbox_backend: SandboxBackend | None = None,
    limits: EpisodeLimits = EpisodeLimits(),
    metadata: dict | None = None,
    token_counter: Callable[[str], int] | None = None,
    dispatcher: Callable | None = None,
    catalog_serializer: Callable = serialize_public_catalog,
) -> EpisodeResult:
    """Run sequentially until final output or a classified terminal condition.

    The caller owns a fresh DomainSession and secure backend. Neither is created
    or reset here. Missing dependencies are classified by dispatch on use, without
    a local execution fallback. token_counter must count the exact context with
    the future provider's template/overhead; it is called before *every* output.
    Bad trusted configuration raises ValueError before any callback or dispatch.
    """
    if (not callable(model_callback) or not isinstance(limits, EpisodeLimits)
            or (token_counter is not None and not callable(token_counter))):
        raise ValueError("Invalid episode configuration.")
    private_metadata = {} if metadata is None else metadata
    if (not isinstance(private_metadata, dict)
            or not set(private_metadata) <= {"run_id", "model_id", "attempt_id"}
            or any(type(v) is not str for v in private_metadata.values())):
        raise ValueError("Invalid execution metadata.")
    dispatcher = dispatch_tool if dispatcher is None else dispatcher
    document = _public_document(task, public_catalog, catalog_serializer)
    result = EpisodeResult(full_history=document, metadata={
        **deepcopy(private_metadata), "task_id": task["id"],
        "phase": "phase2", "condition": "tools",
    })
    context = DispatchContext(domain_session=domain_session, sandbox_backend=sandbox_backend)

    def finish(reason, *, infrastructure=False, error_code=None):
        result.termination_reason = reason
        result.infrastructure_failure = infrastructure
        result.technical_success = result.model_outputs_total > 0 and not infrastructure
        result.error_code = error_code
        result.tool_trace = context.trace.to_dicts()
        result.direct_answer = reason == "final_answer" and result.tool_calls_dispatched == 0
        return result

    while result.model_outputs_total < limits.max_model_outputs:
        # Fixed field order and exact numeric serialization (including Decimal).
        # No dependency, private metadata or trace fields are serialized here.
        try:
            serialized = json_bytes(document).decode("utf-8")
            if token_counter is not None:
                count = token_counter(serialized)
                if type(count) is not int or count < 0:
                    raise ValueError("Invalid token count.")
                result.context_token_counts.append(count)
                result.token_count_validated = True
                if count > limits.input_token_limit:
                    return finish("context_limit_exceeded", error_code="context_limit")
        except Exception:
            return finish("context_preflight_failure", infrastructure=True,
                          error_code="execution_error")

        result.model_contexts.append(serialized)
        try:
            raw = model_callback(serialized)
            if not isinstance(raw, str):
                raise TypeError("Callback must return raw model text.")
        except Exception:
            # Exception messages may contain tokens or paths; never retain them.
            return finish("model_failure", infrastructure=True, error_code="execution_error")
        result.model_outputs_total += 1
        result.model_messages.append(raw)
        try:
            message = parse_model_message(raw)
        except ProtocolError:
            result.protocol_correct = False
            document["history"].append(raw)
            return finish("invalid_protocol", error_code="invalid_protocol")
        result.protocol_correct = True
        document["history"].append(asdict(message))

        if isinstance(message, FinalAnswer):
            result.final_answer = deepcopy(message.answer)
            return finish("final_answer")
        if result.tool_calls_dispatched >= limits.max_tool_calls:
            return finish("tool_call_limit")

        # Unknown names and invalid arguments count; malformed protocol did not
        # reach this point. No ninth call can reach dispatch or change state.
        result.tool_calls_dispatched += 1
        try:
            envelope = dispatcher(message.tool, message.arguments, context)
        except Exception:
            return finish("tool_infrastructure_failure", infrastructure=True,
                          error_code="execution_error")
        result.tool_results.append(deepcopy(envelope))
        document["history"].append(deepcopy(envelope))
        if context.trace.entries[-1].infrastructure_failure:
            return finish("tool_infrastructure_failure", infrastructure=True,
                          error_code=envelope["error"]["code"])

    return finish("model_output_limit")
