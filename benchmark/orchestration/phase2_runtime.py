"""Bridge a loaded Transformers adapter to one existing episode, never a batch run.

Timeout contract: a future runner supplies a 600-second EpisodeDeadline shared
across context building/generation/tools. Checks here are cooperative, before
and after blocking generation, not a hard CUDA interrupt. A hard end-to-end
limit requires a separately supervised worker and confirmed cleanup; this API
never claims to stop an in-flight CUDA kernel. Late output is discarded before
any tool dispatch. Tools retain their own sandbox deadlines/cancellation.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Event
import time
from typing import Callable

from benchmark.providers.transformers_phase2 import EpisodeProviderError, TransformersEpisodeProvider
from .episode import EpisodeLimits, EpisodeResult, run_episode

EPISODE_TIMEOUT_SECONDS = 600


@dataclass
class EpisodeDeadline:
    expires_at: float
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    _cancelled: Event = field(default_factory=Event, repr=False, compare=False)

    @classmethod
    def start(cls, seconds=EPISODE_TIMEOUT_SECONDS, *, clock=time.monotonic):
        if seconds <= 0:
            raise ValueError('Deadline must be positive.')
        return cls(clock() + seconds, clock)

    def request_cancel(self):
        """Cooperative cancellation; never a claim of immediate GPU interruption."""
        self._cancelled.set()

    def check(self, scope):
        if self._cancelled.is_set():
            raise EpisodeProviderError('episode_cancelled', scope)
        if self.clock() >= self.expires_at:
            raise EpisodeProviderError('episode_timeout', scope)


@dataclass
class TransformersEpisodeResult:
    episode: EpisodeResult
    turn_metrics: list
    context_checks: list
    run_metadata: dict


def run_transformers_episode(task, provider: TransformersEpisodeProvider, *,
                             domain_session=None, sandbox_backend=None, public_catalog=None,
                             run_id=None, attempt_id=None, deadline=None, episode_options=None):
    """One sequential task. Caller loads/warms once and owns domain/backend state.

    No model loading, warm-up, fixture reading, evaluation or persistence occurs
    here. Official generation/context settings cannot be overridden per task.
    The provider has no output limit: eight calls/nine outputs belong to episode.
    """
    provider.validate_ready()
    provider.begin_episode(deadline=deadline)
    metadata = {'model_id': provider.model_id}
    if run_id is not None:
        metadata['run_id'] = run_id
    if attempt_id is not None:
        metadata['attempt_id'] = attempt_id
    episode = run_episode(task, provider, token_counter=provider.count_tokens,
                          domain_session=domain_session, sandbox_backend=sandbox_backend,
                          public_catalog=public_catalog, limits=EpisodeLimits(), metadata=metadata,
                          **({} if episode_options is None else episode_options))
    failure = provider.last_failure
    if failure is not None and failure.code in {'episode_timeout', 'episode_cancelled'}:
        # Adapt a cooperative budget stop, without changing the frozen loop or
        # mislabelling it invalid_protocol/provider failure. No late tool action.
        episode.termination_reason = 'timeout'
        episode.error_code = failure.code
        episode.infrastructure_failure = False
        episode.technical_success = (episode.model_outputs_total > 0
                                     and failure.scope == 'between_operations')
    metrics = deepcopy(provider.turn_metrics)
    for metric in metrics:
        if metric.failure is not None:
            metric.failure.update(task_id=task['id'], turn_index=metric.turn_index)
    return TransformersEpisodeResult(episode, metrics,
                                     deepcopy(provider.context_checks), provider.run_metadata())
