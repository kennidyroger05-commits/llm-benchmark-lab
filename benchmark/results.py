"""Provider-independent structure for one benchmark task result."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _utc_timestamp() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BenchmarkResult:
    """Serializable result and measurements for a single inference attempt."""

    task_id: Optional[str] = None
    phase: Optional[str] = None
    condition: Optional[str] = None
    model_id: Optional[str] = None
    provider: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    status: str = "success"
    error: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    do_sample: Optional[bool] = None
    seed: Optional[int] = None

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    model_load_time_seconds: Optional[float] = None
    inference_latency_seconds: Optional[float] = None
    tokens_per_second: Optional[float] = None

    device: Optional[str] = None
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None
    dtype: Optional[str] = None

    timestamp: str = field(default_factory=_utc_timestamp)

    def __post_init__(self):
        # Keep derived measurements consistent when the caller supplies counts/timing.
        if self.status not in {"success", "error"}:
            raise ValueError("status must be 'success' or 'error'")
        if self.total_tokens is None and self.input_tokens is not None and self.output_tokens is not None:
            self.total_tokens = self.input_tokens + self.output_tokens
        if self.tokens_per_second is None and self.output_tokens is not None:
            if self.inference_latency_seconds is not None and self.inference_latency_seconds > 0:
                # This is observed throughput for the complete measured generation call,
                # not an isolated decode-only token rate.
                self.tokens_per_second = self.output_tokens / self.inference_latency_seconds

    def to_dict(self) -> dict:
        """Return a plain dictionary containing only serializable values."""
        return asdict(self)
