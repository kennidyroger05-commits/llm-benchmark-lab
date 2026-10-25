"""Phase 2 adapter around the unchanged Phase 1 TransformersProvider.

A prepared-tokenizer facade lets preflight and generation consume the *same*
encoded inputs: the real chat template and tokenizer run once per turn, while
Phase 1 still owns no_grad, CUDA timing, generation, suffix slicing and decoding.
No loading occurs on import/construction; HF authentication remains implicit in
from_pretrained (HF_TOKEN or an existing login), never copied into metadata.
Single-caller/sequential API, like the existing episode and Phase 1 runner.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import platform
from types import MappingProxyType, SimpleNamespace

from .transformers_provider import TransformersProvider
from .failure_diagnostics import failure_details
from .phase2_memory import (configure_memory_attention, efficient_cuda_kernels,
                            cuda_memory_observation, execution_metadata, observe_forward)
from benchmark.orchestration.context_diagnostics import context_sizes

OFFICIAL_MODELS = (
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
)
GENERATION_OPTIONS = MappingProxyType({
    'max_new_tokens': 512, 'do_sample': False, 'temperature': None, 'seed': None,
})
CONTEXT_TOKENS = 16384
INPUT_TOKEN_LIMIT = CONTEXT_TOKENS - GENERATION_OPTIONS['max_new_tokens']
WARMUP_PROMPTS = ('Responda apenas: pronto.', 'Quanto é 2 + 2? Responda apenas com o número.')


class EpisodeProviderError(RuntimeError):
    """Stable, sanitized provider diagnostic; never contains a loader exception."""
    def __init__(self, code, scope='provider', *, diagnostic=None):
        self.code, self.scope = code, scope
        self.diagnostic = diagnostic
        super().__init__('Phase 2 provider: ' + code)


@dataclass(frozen=True)
class ContextCheck:
    turn_index: int
    document_text: str
    tokenizer_text: str
    input_tokens: int
    input_token_limit: int
    context_limit: bool
    declared_capacity: int
    context_components: dict | None = None


@dataclass(frozen=True)
class TurnMetrics:
    turn_index: int
    input_tokens: int
    output_tokens: int | None
    inference_latency_seconds: float | None
    hit_max_new_tokens: bool | None
    raw_output: str | None
    device: str | None
    dtype: str | None
    status: str
    error_code: str | None = None
    failure: dict | None = None
    cuda_memory: dict | None = None
    max_new_tokens: int = 512
    execution: dict | None = None


class _PreparedTensor:
    """Observe placement errors before the Phase 1 core converts them to text."""
    def __init__(self, tensor, observe):
        self.tensor, self.observe = tensor, observe

    @property
    def shape(self):
        return self.tensor.shape

    def to(self, device):
        try:
            return self.tensor.to(device)
        except Exception as error:
            self.observe(error, 'input_placement')
            raise


class _PreparedTokenizer:
    """A facade, not a second tokenizer/template. Owns one detached turn cache."""
    def __init__(self, tokenizer, observe):
        self.raw = tokenizer
        self.observe = observe
        self.clear()

    @property
    def chat_template(self):
        return self.raw.chat_template

    def clear(self):
        self.document = self.text = self.inputs = None

    def prepare(self, document):
        if self.document != document:
            text = self.raw.apply_chat_template(
                [{'role': 'user', 'content': document}], tokenize=False, add_generation_prompt=True)
            inputs = self.raw(text, return_tensors='pt', add_special_tokens=False, truncation=False)
            self.document, self.text, self.inputs = document, text, inputs
        return self.inputs['input_ids'].shape[-1]

    def apply_chat_template(self, messages, **options):
        if messages != [{'role': 'user', 'content': self.document}] or options != {
                'tokenize': False, 'add_generation_prompt': True}:
            raise EpisodeProviderError('prepared_input_mismatch')
        return self.text

    def __call__(self, text, **options):
        if text != self.text or options != {'return_tensors': 'pt', 'add_special_tokens': False}:
            raise EpisodeProviderError('prepared_input_mismatch')
        return {key: _PreparedTensor(value, self.observe) for key, value in self.inputs.items()}

    def decode(self, *args, **kwargs):
        try:
            return self.raw.decode(*args, **kwargs)
        except Exception as error:
            self.observe(error, 'decode')
            raise


class _ExecutionModel:
    """Expose Accelerate's actual embedding execution device to Phase 1 logic.

    Offloaded embedding storage may be on CPU/meta while its execution hook uses
    CUDA. This proxy changes neither weights nor hooks, and never guesses a CPU
    fallback. The real model still performs generation and device-map dispatch.
    """
    def __init__(self, model, device, observe, evidence):
        self.raw, self.device = model, device
        self.observe = observe
        self.evidence = evidence

    def get_input_embeddings(self):
        weight = self.raw.get_input_embeddings().weight
        return SimpleNamespace(weight=SimpleNamespace(device=self.device, dtype=weight.dtype))

    def generate(self, **options):
        try:
            with observe_forward(self.raw, self.evidence), efficient_cuda_kernels(self.device):
                return self.raw.generate(**options)
        except Exception as error:
            # The unchanged Phase 1 core returns only str(error). Preserve its
            # type and stage here, before that conversion, without retrying.
            self.observe(error, 'model_generate')
            raise


class TransformersEpisodeProvider:
    """Explicit loading, real token preflight and a callable raw-text adapter."""
    def __init__(self, model_id, *, provider=None):
        if model_id not in OFFICIAL_MODELS:
            raise ValueError('Unsupported official model ID.')
        self.model_id = model_id
        self._core = (TransformersProvider(model_id, 'float16', 'auto') if provider is None else provider)
        if (self._core.model_id != model_id or self._core.device_map != 'auto'
                or str(self._core.torch_dtype).replace('torch.', '') != 'float16'):
            raise ValueError('Phase 2 requires the official FP16/auto configuration.')
        self._tokenizer = None
        self.turn_metrics = []
        self.context_checks = []
        self._all_turns = []
        self.last_failure = None
        self._generation_failure = None
        self.deadline = None
        self.warmup_completed = False
        self.attention_implementation_before = None
        self._execution_evidence = {}

    def load_tokenizer(self):
        try:
            raw = self._core.load_tokenizer()
            self._tokenizer = _PreparedTokenizer(raw, self._observe_failure)
            self._core.tokenizer = self._tokenizer
        except Exception as error:
            raise EpisodeProviderError('tokenizer_load_failure',
                diagnostic=failure_details(error, stage='tokenizer_load')) from None
        return raw

    def load_model(self):
        import torch
        # Exactly the Phase 1 loader, dtype/device_map and load-time measurement.
        self._core.torch_dtype = torch.float16
        try:
            model = self._core.load_model()
            self.attention_implementation_before = configure_memory_attention(model)
            return model
        except Exception as error:
            raise EpisodeProviderError('model_load_failure',
                diagnostic=failure_details(error, stage='model_load')) from None

    def _ready_tokenizer(self):
        if self._core.model is None or self._core.tokenizer is None:
            raise EpisodeProviderError('not_loaded')
        if self._tokenizer is None:
            self._tokenizer = _PreparedTokenizer(self._core.tokenizer, self._observe_failure)
            self._core.tokenizer = self._tokenizer
        if not self._tokenizer.chat_template:
            raise EpisodeProviderError('chat_template_unavailable')
        return self._tokenizer

    def declared_capacity(self):
        """Use declared model/tokenizer bounds; ignore HF's huge unknown sentinel."""
        tokenizer = self._ready_tokenizer().raw
        config = getattr(self._core.model, 'config', None)
        values = [getattr(config, 'max_position_embeddings', None),
                  getattr(getattr(config, 'text_config', None), 'max_position_embeddings', None),
                  getattr(tokenizer, 'model_max_length', None)]
        capacities = [v for v in values if type(v) is int and 0 < v < 10**9]
        if not capacities:
            raise EpisodeProviderError('context_capacity_unverified', 'preflight')
        return min(capacities)

    def validate_ready(self):
        """Future official preflight must call this before warm-up/any task."""
        capacity = self.declared_capacity()
        if capacity < CONTEXT_TOKENS:
            raise EpisodeProviderError('context_capacity_insufficient', 'preflight')
        if str(self._core.model.get_input_embeddings().weight.dtype).replace('torch.', '') != 'float16':
            raise EpisodeProviderError('dtype_mismatch', 'preflight')
        return capacity

    def _check_deadline(self, scope):
        if self.deadline is not None:
            self.deadline.check(scope)

    def count_tokens(self, document):
        """Real post-template count. A limit violation is recorded, not truncated."""
        try:
            self._check_deadline('between_operations')
            capacity = self.validate_ready()
            tokenizer = self._ready_tokenizer()
            count = tokenizer.prepare(document)
            if type(count) is not int or count < 0:
                raise EpisodeProviderError('invalid_token_count', 'preflight')
            check = ContextCheck(len(self.turn_metrics) + 1, document, tokenizer.text, count,
                                 INPUT_TOKEN_LIMIT, count > INPUT_TOKEN_LIMIT, capacity,
                                 context_sizes(document))
            # callback consumes the preflight cache, without counting a second turn.
            if not self.context_checks or self.context_checks[-1] != check:
                self.context_checks.append(check)
            return count
        except EpisodeProviderError as error:
            self.last_failure = error
            raise
        except Exception as error:
            self.last_failure = EpisodeProviderError('tokenization_failure', 'preflight',
                diagnostic=failure_details(error, stage='tokenization'))
            raise self.last_failure from None

    def _observe_failure(self, error, stage):
        self._generation_failure = failure_details(error, stage=stage)

    def _input_device(self):
        model = self._core.model
        embeddings = model.get_input_embeddings()
        def hook_device(hook):
            found = getattr(hook, 'execution_device', None)
            if found is not None:
                return found
            # Accelerate can compose AlignDevices/offload hooks in SequentialHook.
            for child in getattr(hook, 'hooks', ()):
                found = hook_device(child)
                if found is not None:
                    return found
            return None
        for module in (embeddings, model):
            device = hook_device(getattr(module, '_hf_hook', None))
            if device is not None:
                import torch
                return torch.device('cuda:' + str(device) if type(device) is int else device)
        device = embeddings.weight.device
        if str(device) == 'meta':
            raise EpisodeProviderError('execution_device_unavailable')
        return device

    @contextmanager
    def _execution_model(self):
        original = self._core.model
        self._core.model = _ExecutionModel(original, self._input_device(), self._observe_failure, self._execution_evidence)
        try:
            yield
        finally:
            self._core.model = original

    def __call__(self, document):
        """Callback compatible with run_episode; never parses or repairs output."""
        return self._generate(document, GENERATION_OPTIONS['max_new_tokens'])

    def generate_diagnostic(self, document, *, max_new_tokens=8):
        """Non-benchmark entry: same generation, smaller explicit output budget."""
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8:
            raise ValueError('Diagnostic output budget must be between 1 and 8 tokens.')
        return self._generate(document, max_new_tokens)

    def _generate(self, document, max_new_tokens):
        self._generation_failure = None
        self._execution_evidence = {}
        memory = None
        try:
            count = self.count_tokens(document)
            if count > INPUT_TOKEN_LIMIT:
                raise EpisodeProviderError('context_limit', 'preflight')
            self._check_deadline('between_operations')
            with self._execution_model():
                # Reuse Phase 1 hardware collection, timing, sync, suffix and decode.
                self._core.hardware = self._core._collect_hardware_metadata()
                with cuda_memory_observation(self._core.model.device) as memory:
                    result = self._core.generate(document, **{**GENERATION_OPTIONS, 'max_new_tokens': max_new_tokens})
            if result.status != 'success':
                diagnostic = self._generation_failure
                if diagnostic is None:
                    # A custom core can return an error without raising one;
                    # do not invent the original exception's class in that case.
                    diagnostic = failure_details(RuntimeError(result.error or 'Provider returned an error.'),
                                                 stage='provider_generate')
                    diagnostic['exception_type'] = 'ProviderResultError'
                raise EpisodeProviderError('generation_failure', diagnostic=diagnostic)
            # A future runner must not dispatch actions after an expired generation.
            self._check_deadline('generation')
            metric = TurnMetrics(len(self.turn_metrics) + 1, count, result.output_tokens,
                                 result.inference_latency_seconds, result.output_tokens == max_new_tokens,
                                 result.response, result.device, result.dtype, 'success',
                                 cuda_memory=memory, max_new_tokens=max_new_tokens, execution=self._execution_evidence)
        except EpisodeProviderError as error:
            self.last_failure = error
            # Context/capacity refusals are preflight evidence, not generated turns.
            if error.scope != 'preflight' and 'count' in locals():
                attempted = locals().get('result')
                metric = TurnMetrics(len(self.turn_metrics) + 1, count,
                                     getattr(attempted, 'output_tokens', None),
                                     getattr(attempted, 'inference_latency_seconds', None), None,
                                     None, self._core.hardware.get('device'),
                                     self._core.hardware.get('dtype'), 'error', error.code,
                                     error.diagnostic, memory, max_new_tokens, self._execution_evidence)
                self.turn_metrics.append(metric)
                self._all_turns.append(metric)
            raise
        except Exception as error:
            self.last_failure = EpisodeProviderError('generation_failure',
                diagnostic=failure_details(error, stage='provider_generate'))
            if 'count' in locals():
                metric = TurnMetrics(len(self.turn_metrics) + 1, count, None, None, None, None,
                    self._core.hardware.get('device'), self._core.hardware.get('dtype'),
                    'error', 'generation_failure', self.last_failure.diagnostic, memory, max_new_tokens, self._execution_evidence)
                self.turn_metrics.append(metric)
                self._all_turns.append(metric)
            raise self.last_failure from None
        finally:
            # Tokenizer inputs remain on CPU, but release the cache on errors too.
            if self._tokenizer is not None:
                self._tokenizer.clear()
        self.turn_metrics.append(metric)
        self._all_turns.append(metric)
        # Clear only after consumption so identical successive turns are tokenized
        # afresh, while preflight + callback still share one preparation.
        return result.response

    def begin_episode(self, *, deadline=None):
        """Fresh metrics/cache, never a model-output limit (owned solely by episode)."""
        self.turn_metrics = []
        self.context_checks = []
        self.last_failure = None
        self.deadline = deadline
        if self._tokenizer is not None:
            self._tokenizer.clear()

    def warm_up(self):
        """Exactly two independent short calls; discard all their execution metrics."""
        self.validate_ready()
        if self._all_turns or self.turn_metrics or self.warmup_completed:
            raise EpisodeProviderError('warmup_requires_fresh_provider', 'preflight')
        try:
            for prompt in WARMUP_PROMPTS:
                self.begin_episode()
                self(prompt)
            import torch
            device = self._input_device()
            if str(device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            self.warmup_completed = True
        finally:
            self._all_turns = []
            self.begin_episode()

    def run_metadata(self):
        """Allowlisted provenance/hardware; never serialize full HF config/env."""
        tokenizer = self._ready_tokenizer().raw
        with self._execution_model():
            hardware = self._core._collect_hardware_metadata()
        import torch
        device_map = {str(k): ('cuda:' + str(v) if type(v) is int else str(v))
                      for k, v in getattr(self._core.model, 'hf_device_map', {}).items()}
        cuda_used = hardware['device'].startswith('cuda') or any(v.startswith('cuda') for v in device_map.values())
        offloaded = sorted(k for k, v in device_map.items() if v == 'disk' or (v == 'cpu' and cuda_used))
        def package(name):
            try:
                return version(name)
            except PackageNotFoundError:
                return None
        config = getattr(self._core.model, 'config', None)
        return {
            'model_id': self.model_id, 'model_revision': getattr(config, '_commit_hash', None),
            'tokenizer_revision': getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash'),
            'python_version': platform.python_version(),
            **{name + '_version': package(name) for name in ('transformers', 'torch', 'accelerate')},
            'model_load_time_seconds': self._core.model_load_time_seconds,
            **hardware, 'cuda_available': torch.cuda.is_available(),
            'device_map': dict(sorted(device_map.items())), 'offloaded_modules': offloaded,
            'offload_observed': bool(offloaded), 'warmup_completed': self.warmup_completed,
            'total_input_tokens': sum(t.input_tokens for t in self._all_turns),
            'total_output_tokens': sum(t.output_tokens or 0 for t in self._all_turns),
            'total_generation_latency_seconds': sum(t.inference_latency_seconds or 0 for t in self._all_turns),
            'successful_turns': sum(t.status == 'success' for t in self._all_turns),
            'failed_turns': sum(t.status == 'error' for t in self._all_turns),
            **execution_metadata(self._core.model),
            'attention_implementation_before': self.attention_implementation_before,
            'generation_kwargs': {'max_new_tokens': 512, 'do_sample': False},
            'last_forward': self._execution_evidence.copy(),
        }
