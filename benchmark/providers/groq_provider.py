"""Groq implementation of the shared inference provider interface."""

import os

from dotenv import load_dotenv
from groq import APIError, Groq

from benchmark.providers.base import LLMProvider


class GroqProvider(LLMProvider):
    """Discover available Groq models using the official SDK."""

    def __init__(self):
        # Load values from a local .env file without overriding environment values.
        load_dotenv()

        # Validate credentials before creating the SDK client.
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "A variável GROQ_API_KEY não foi encontrada. "
                "Crie um arquivo .env baseado no .env.example e adicione sua chave."
            )

        # The official SDK uses this client for authenticated Groq API requests.
        self.client = Groq(api_key=api_key)

    def get_available_models(self) -> list[str]:
        """Return the IDs of models available to the configured Groq account."""
        try:
            response = self.client.models.list()
        except APIError as error:
            # Keep provider errors readable without exposing a Python traceback.
            raise RuntimeError(f"Falha ao consultar a API da Groq: {error}") from None

        # Return only model identifiers so callers can format the list as needed.
        return [model.id for model in response.data]
