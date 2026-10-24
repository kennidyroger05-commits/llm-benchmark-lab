"""Strict, complete JSON message parsing, independent of catalog and tasks."""
from dataclasses import dataclass, field
from decimal import Decimal, DecimalException
import json
from typing import Literal, Union

# Integer literals stay int. Fraction/exponent literals use Decimal, never float.
JSONValue = Union[
    None, bool, int, Decimal, str, list["JSONValue"], dict[str, "JSONValue"]
]

@dataclass(frozen=True)
class ToolCall:
    """A valid envelope; existence of the tool is a future dispatcher concern."""
    tool: str
    arguments: dict[str, JSONValue]
    type: Literal["tool_call"] = field(default="tool_call", init=False)

@dataclass(frozen=True)
class FinalAnswer:
    """An unmodified JSON answer, without task-specific format validation."""
    answer: JSONValue
    type: Literal["final_answer"] = field(default="final_answer", init=False)

class ProtocolError(ValueError):
    """Invalid protocol message; callers can classify by code."""
    code = "invalid_protocol"

def _unique_object(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    # Called for every object, including nested arguments and answers.
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("Duplicate JSON object key.")
        result[key] = value
    return result

def _reject_constant(value: str):
    raise ProtocolError(f"Non-JSON numeric constant: {value}.")

def _parse_decimal(value: str) -> Decimal:
    number = Decimal(value)
    if not number.is_finite():
        raise ProtocolError("JSON numbers must be finite.")
    return number

def parse_model_message(raw: str) -> Union[ToolCall, FinalAnswer]:
    """Parse the whole text, accepting JSON whitespace but never extracting/repairing.

    Integers remain int; fractional/exponent numbers become exact Decimal values.
    This preserves numeric precision without converting numbers into strings.
    """
    if not isinstance(raw, str):
        raise ProtocolError("Model output must be a string.")
    try:
        message = json.loads(
            raw, object_pairs_hook=_unique_object,
            parse_float=_parse_decimal, parse_constant=_reject_constant,
        )
    except ProtocolError:
        raise
    except (ValueError, TypeError, DecimalException, RecursionError) as error:
        raise ProtocolError("Expected one complete JSON document.") from error
    if not isinstance(message, dict):
        raise ProtocolError("Model message must be a JSON object.")
    message_type = message.get("type")
    if message_type == "tool_call":
        if set(message) != {"type", "tool", "arguments"}:
            raise ProtocolError("tool_call requires exactly type, tool and arguments.")
        if not isinstance(message["tool"], str) or not message["tool"]:
            raise ProtocolError("tool must be a non-empty string.")
        if not isinstance(message["arguments"], dict):
            raise ProtocolError("arguments must be a JSON object.")
        return ToolCall(tool=message["tool"], arguments=message["arguments"])
    if message_type == "final_answer":
        if set(message) != {"type", "answer"}:
            raise ProtocolError("final_answer requires exactly type and answer.")
        return FinalAnswer(answer=message["answer"])
    raise ProtocolError("Unknown or missing message type.")
