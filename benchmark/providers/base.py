"""Minimal interface shared by inference providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base interface for model generation and provider model discovery."""

    def generate(self, prompt: str, **kwargs):
        """Generate text from a prompt; concrete providers can implement this later."""
        raise NotImplementedError("Text generation is not implemented yet.")

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Return IDs for the models available from this provider."""
        raise NotImplementedError
