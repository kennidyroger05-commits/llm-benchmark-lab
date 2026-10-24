"""Small helpers for connecting to the Groq API."""

import os

from dotenv import load_dotenv
from groq import Groq
from groq import APIError


def list_available_models():
    """Return the IDs of models available to the configured Groq account."""
    # Load local .env values while allowing explicitly configured environment values.
    load_dotenv()

    # Fail with an actionable message before constructing the SDK client.
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "A variável GROQ_API_KEY não foi encontrada. "
            "Crie um arquivo .env baseado no .env.example e adicione sua chave."
        )

    # The official SDK uses the key to authenticate the models list request.
    client = Groq(api_key=api_key)
    try:
        response = client.models.list()
    except APIError as error:
        # Keep provider errors readable without exposing a Python traceback.
        raise RuntimeError(f"Falha ao consultar a API da Groq: {error}") from None

    # Return only stable model identifiers for concise display by the entry point.
    return [model.id for model in response.data]
