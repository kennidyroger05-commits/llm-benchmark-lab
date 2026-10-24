"""Smoke tests for the Transformers provider without downloading model files."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmark.providers.base import LLMProvider
from benchmark.providers.transformers_provider import TransformersProvider


class TransformersProviderStructureTests(unittest.TestCase):
    """Check provider structure and its explicit loading boundary."""

    def test_provider_starts_unloaded_and_lists_configured_model(self):
        # Constructing the provider must not import or load model artifacts.
        provider = TransformersProvider(
            model_id="example/model",
            torch_dtype="auto",
            device_map="auto",
        )

        self.assertIsInstance(provider, LLMProvider)
        self.assertIsNone(provider.tokenizer)
        self.assertIsNone(provider.model)
        self.assertEqual(provider.get_available_models(), ["example/model"])

    def test_generate_returns_error_result_when_model_is_not_loaded(self):
        # An unloaded provider reports a common result instead of starting inference.
        provider = TransformersProvider("example/model", "auto", "auto")

        result = provider.generate("Hello")
        self.assertEqual(result.status, "error")
        self.assertIn("load_tokenizer()", result.error)

    def test_generate_returns_token_and_latency_metrics_on_cpu(self):
        class FakeTensor(list):
            @property
            def shape(self):
                return (1, len(self))

            def to(self, device):
                return self

        class FakeTokenizer:
            chat_template = None

            def __call__(self, prompt, **kwargs):
                return {"input_ids": FakeTensor([10, 11, 12])}

            def decode(self, tokens, **kwargs):
                self.decoded_tokens = tokens
                self.decode_options = kwargs
                return "answer"

        class FakeModel:
            def get_input_embeddings(self):
                return SimpleNamespace(weight=SimpleNamespace(device="cpu", dtype="float16"))

            def generate(self, **kwargs):
                self.options = kwargs
                return [FakeTensor([10, 11, 12, 20, 21])]

        provider = TransformersProvider("example/model", "float16", "auto")
        provider.tokenizer = FakeTokenizer()
        provider.model = FakeModel()
        provider.model_load_time_seconds = 1.25
        provider.hardware = {"device": "cpu", "dtype": "float16"}
        synchronize = Mock()
        manual_seed = Mock()
        torch_stub = SimpleNamespace(
            no_grad=nullcontext,
            manual_seed=manual_seed,
            cuda=SimpleNamespace(is_available=lambda: False, synchronize=synchronize),
        )

        with patch.dict("sys.modules", {"torch": torch_stub}):
            result = provider.generate("question", max_new_tokens=16)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.response, "answer")
        self.assertEqual(provider.tokenizer.decoded_tokens, [20, 21])
        self.assertFalse(provider.tokenizer.decode_options["clean_up_tokenization_spaces"])
        self.assertEqual(provider.model.options["max_new_tokens"], 16)
        self.assertFalse(provider.model.options["do_sample"])
        self.assertNotIn("temperature", provider.model.options)
        self.assertNotIn("generator", provider.model.options)
        synchronize.assert_not_called()
        manual_seed.assert_not_called()
        self.assertEqual(result.input_tokens, 3)
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.total_tokens, 5)
        self.assertGreater(result.inference_latency_seconds, 0)
        self.assertAlmostEqual(result.tokens_per_second, 2 / result.inference_latency_seconds)
        self.assertEqual(result.model_load_time_seconds, 1.25)
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.dtype, "float16")
        self.assertIsNone(result.temperature)
        self.assertIsNone(result.seed)

    def test_cuda_generation_synchronizes_before_and_after_measurement(self):
        class FakeTensor(list):
            @property
            def shape(self):
                return (1, len(self))

            def to(self, device):
                return self

        class FakeTokenizer:
            chat_template = None

            def __call__(self, prompt, **kwargs):
                return {"input_ids": FakeTensor([1])}

            def decode(self, tokens, **kwargs):
                return "ok"

        class FakeModel:
            def get_input_embeddings(self):
                return SimpleNamespace(weight=SimpleNamespace(device="cuda:0", dtype="float16"))

            def generate(self, **kwargs):
                self.options = kwargs
                return [FakeTensor([1, 2])]

        synchronize = Mock()
        manual_seed = Mock()
        torch_stub = SimpleNamespace(
            no_grad=nullcontext,
            manual_seed=manual_seed,
            cuda=SimpleNamespace(is_available=lambda: True, synchronize=synchronize),
        )
        provider = TransformersProvider("example/model", "float16", "auto")
        provider.model = FakeModel()
        provider.tokenizer = FakeTokenizer()

        with patch.dict("sys.modules", {"torch": torch_stub}):
            result = provider.generate("question", do_sample=True, temperature=0.6, seed=7)

        self.assertEqual(result.status, "success")
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(synchronize.call_args_list[0].args, ("cuda:0",))
        self.assertEqual(synchronize.call_args_list[1].args, ("cuda:0",))
        manual_seed.assert_called_once_with(7)
        self.assertEqual(result.temperature, 0.6)
        self.assertTrue(result.do_sample)
        self.assertEqual(result.seed, 7)
        self.assertEqual(result.max_new_tokens, 128)
        self.assertEqual(provider.model.options["temperature"], 0.6)
        self.assertNotIn("generator", provider.model.options)

    def test_hardware_metadata_works_without_cuda(self):
        class FakeModel:
            def get_input_embeddings(self):
                return SimpleNamespace(weight=SimpleNamespace(device="cpu", dtype="float32"))

        provider = TransformersProvider("example/model", "auto", "auto")
        provider.model = FakeModel()
        torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

        with patch.dict("sys.modules", {"torch": torch_stub}):
            metadata = provider._collect_hardware_metadata()

        self.assertEqual(metadata, {"device": "cpu", "dtype": "float32"})

    def test_load_model_uses_dtype_and_records_elapsed_time(self):
        model = SimpleNamespace(eval=lambda: None)
        loader = Mock(return_value=model)
        transformers_stub = SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader)
        )
        provider = TransformersProvider("example/model", "float16", "auto")
        provider._collect_hardware_metadata = lambda: {"device": "cpu", "dtype": "float16"}

        with patch.dict("sys.modules", {"transformers": transformers_stub}):
            loaded_model = provider.load_model()

        loader.assert_called_once_with("example/model", dtype="float16", device_map="auto")
        self.assertIs(loaded_model, model)
        self.assertGreaterEqual(provider.model_load_time_seconds, 0)
        self.assertEqual(provider.hardware["device"], "cpu")

    def test_generation_exception_is_returned_as_error(self):
        class FakeTensor(list):
            @property
            def shape(self):
                return (1, len(self))

            def to(self, device):
                return self

        class FakeTokenizer:
            chat_template = None

            def __call__(self, prompt, **kwargs):
                return {"input_ids": FakeTensor([1])}

        class FakeModel:
            def get_input_embeddings(self):
                return SimpleNamespace(weight=SimpleNamespace(device="cpu", dtype="float32"))

            def generate(self, **kwargs):
                raise ValueError("generation failed")

        provider = TransformersProvider("example/model", "auto", "auto")
        provider.tokenizer = FakeTokenizer()
        provider.model = FakeModel()
        torch_stub = SimpleNamespace(no_grad=nullcontext)

        with patch.dict("sys.modules", {"torch": torch_stub}):
            result = provider.generate("question")

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, "generation failed")
        self.assertEqual(result.input_tokens, 1)


if __name__ == "__main__":
    unittest.main()
