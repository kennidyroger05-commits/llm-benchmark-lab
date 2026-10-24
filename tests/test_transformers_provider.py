"""Smoke tests for the Transformers provider without downloading model files."""

import unittest

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

    def test_generate_requires_explicit_loading(self):
        # Generation must fail clearly before either component has been loaded.
        provider = TransformersProvider("example/model", "auto", "auto")

        with self.assertRaisesRegex(RuntimeError, r"load_tokenizer\(\).*load_model\(\)"):
            provider.generate("Hello")


if __name__ == "__main__":
    unittest.main()
