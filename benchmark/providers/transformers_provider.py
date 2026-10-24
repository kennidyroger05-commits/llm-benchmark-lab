"""Local causal language model provider backed by Hugging Face Transformers."""

from benchmark.providers.base import LLMProvider


class TransformersProvider(LLMProvider):
    """Load and generate text with a configured local Transformers model."""

    def __init__(self, model_id: str, torch_dtype, device_map):
        # Keep configuration only; loading model weights is always explicit.
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.tokenizer = None
        self.model = None

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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        # Put the model into inference mode after loading it.
        self.model.eval()
        return self.model

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate a continuation and return only newly generated text."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer e modelo não carregados. "
                "Chame load_tokenizer() e load_model() antes de generate()."
            )

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

        # Place inputs with the model's input embeddings for CPU or CUDA execution.
        input_device = self.model.get_input_embeddings().weight.device
        encoded_inputs = {
            name: tensor.to(input_device) for name, tensor in encoded_inputs.items()
        }

        # Disable gradient tracking because this provider is intended for inference.
        import torch

        with torch.no_grad():
            generated = self.model.generate(
                **encoded_inputs,
                max_new_tokens=max_new_tokens,
            )

        # Decoder-only generation begins with the input tokens; discard that prefix.
        input_length = encoded_inputs["input_ids"].shape[1]
        new_tokens = generated[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_available_models(self) -> list[str]:
        """Return the configured model ID without downloading or loading it."""
        return [self.model_id]
