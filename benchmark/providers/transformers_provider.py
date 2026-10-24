"""Local causal language model provider backed by Hugging Face Transformers."""

import time

from benchmark.providers.base import LLMProvider
from benchmark.results import BenchmarkResult


class TransformersProvider(LLMProvider):
    """Load and generate text with a configured local Transformers model."""

    def __init__(self, model_id: str, torch_dtype, device_map):
        # Keep configuration only; loading model weights is always explicit.
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.tokenizer = None
        self.model = None
        self.model_load_time_seconds = None
        self.hardware = {}

    def load_tokenizer(self):
        """Load the tokenizer separately from the model weights."""
        # Import lazily so importing this provider does not require model setup.
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self.tokenizer

    def load_model(self):
        """Load model weights with the configured dtype and device placement."""
        # Import lazily so no Transformers or PyTorch work happens at module import.
        from transformers import AutoModelForCausalLM

        start_time = time.perf_counter()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            # Put the model into inference mode after loading it.
            model.eval()
            self.model = model
            self.hardware = self._collect_hardware_metadata()
        finally:
            # Save load duration independently so later results can reuse it.
            self.model_load_time_seconds = time.perf_counter() - start_time
        return self.model

    def _collect_hardware_metadata(self) -> dict:
        """Collect model device/dtype and CUDA details when a GPU is available."""
        import torch

        weight = self.model.get_input_embeddings().weight
        device = str(weight.device)
        metadata = {"device": device, "dtype": str(weight.dtype).replace("torch.", "")}

        if device.startswith("cuda") and torch.cuda.is_available():
            device_index = weight.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            metadata["gpu_name"] = torch.cuda.get_device_name(device_index)
            metadata["vram_gb"] = properties.total_memory / (1024 ** 3)
        return metadata

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = None,
        do_sample: bool = False,
        seed: int = None,
    ) -> BenchmarkResult:
        """Generate text and return its output together with inference measurements."""
        # Temperature and seed have no effect in greedy decoding, so record None there.
        effective_temperature = temperature if do_sample else None
        effective_seed = seed if do_sample else None
        metadata = {
            "model_id": self.model_id,
            "provider": "transformers",
            "prompt": prompt,
            "model_load_time_seconds": self.model_load_time_seconds,
            "max_new_tokens": max_new_tokens,
            "temperature": effective_temperature,
            "do_sample": do_sample,
            "seed": effective_seed,
            **self.hardware,
        }
        if self.model is None or self.tokenizer is None:
            return BenchmarkResult(
                status="error",
                error="Tokenizer e modelo não carregados. Chame load_tokenizer() e load_model() antes de generate().",
                **metadata,
            )

        import torch

        input_tokens = None
        inference_start = None
        try:
            # Use the tokenizer's chat format when available; otherwise pass plain text.
            has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
            if has_chat_template:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted_prompt = prompt

            # Avoid duplicating special tokens already included by a chat template.
            encoded_inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=not has_chat_template,
            )
            input_tokens = encoded_inputs["input_ids"].shape[-1]

            # Place inputs with the model's input embeddings for CPU or CUDA execution.
            input_device = self.model.get_input_embeddings().weight.device
            cuda_in_use = str(input_device).startswith("cuda") and torch.cuda.is_available()
            encoded_inputs = {
                name: tensor.to(input_device) for name, tensor in encoded_inputs.items()
            }

            # Measure generation only; tokenization and decoding are outside this interval.
            if effective_seed is not None:
                # Seed PyTorch before sampling without passing a Generator to generate().
                torch.manual_seed(effective_seed)
            if cuda_in_use:
                torch.cuda.synchronize(input_device)
            inference_start = time.perf_counter()
            generation_options = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
            if effective_temperature is not None:
                generation_options["temperature"] = effective_temperature
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded_inputs,
                    **generation_options,
                )
            if cuda_in_use:
                torch.cuda.synchronize(input_device)
            latency = time.perf_counter() - inference_start

            # Decoder-only generation begins with input tokens; count/decode only the suffix.
            new_tokens = generated[0][input_tokens:]
            output_tokens = len(new_tokens)
            response = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return BenchmarkResult(
                response=response,
                status="success",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                inference_latency_seconds=latency,
                **metadata,
            )
        except Exception as error:
            # Keep inference failures in the common result format for the caller.
            latency = time.perf_counter() - inference_start if inference_start is not None else None
            return BenchmarkResult(
                status="error",
                error=str(error),
                input_tokens=input_tokens,
                inference_latency_seconds=latency,
                **metadata,
            )

    def get_available_models(self) -> list[str]:
        """Return the configured model ID without downloading or loading it."""
        return [self.model_id]
