"""Check the Groq API connection without running a benchmark."""

from benchmark.providers.groq_provider import GroqProvider


def main():
    """Print whether Groq is reachable and list the available model IDs."""
    try:
        # Keep this entry point focused on connection validation for now.
        model_ids = GroqProvider().get_available_models()
    except (ValueError, RuntimeError) as error:
        print(error)
        return 1

    print("Conexão com Groq: OK")
    print("\nModelos disponíveis:")
    for model_id in model_ids:
        print(f"- {model_id}")
    print(f"\nTotal de modelos encontrados: {len(model_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
