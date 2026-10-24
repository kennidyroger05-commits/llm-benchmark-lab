"""Detached structural call trace: no clocks, private sessions or expected answers."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from benchmark.tools._sandbox_worker import json_bytes, pack


def finite_json_bytes(value):
    """Canonical object keys, stable array order, exact Decimal numeric tokens."""
    pack(value)  # Validate finite JSON before sorting or recursively copying.
    def canonical(v):
        if type(v) is dict:
            return {key: canonical(v[key]) for key in sorted(v)}
        if type(v) is list:
            return [canonical(child) for child in v]
        return v
    return json_bytes(canonical(value))


@dataclass(frozen=True)
class ToolTraceEntry:
    call_index: int
    tool: str | None
    arguments: Any
    envelope: dict
    dispatched: bool
    successful: bool
    failed: bool
    error_code: str | None
    dispatch_completed: bool
    tool_known: bool
    tool_execution_completed: bool
    infrastructure_failure: bool
    arguments_recorded: bool = True
    error_origin: str | None = None


class ToolTrace:
    """One controller-owned trace per context; returned snapshots cannot edit history."""

    def __init__(self):
        self.__entries = []

    @property
    def entries(self):
        return tuple(deepcopy(self.__entries))

    def append(self, **fields):
        entry = ToolTraceEntry(call_index=len(self.__entries) + 1, **deepcopy(fields))
        self.__entries.append(entry)
        return deepcopy(entry)

    def to_dicts(self):
        return [asdict(entry) for entry in self.entries]

    def serialize(self):
        """Exact finite JSON UTF-8, including Decimal numbers, in dispatch order."""
        return finite_json_bytes(self.to_dicts())
