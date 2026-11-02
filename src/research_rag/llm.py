from langchain_core.language_models import BaseChatModel


def build_chat_model(model: str) -> BaseChatModel:
    """An OpenAI model when the name starts with 'gpt', otherwise a local Ollama model."""
    if model.startswith("gpt"):
        from langchain_openai import ChatOpenAI

        # No temperature: current OpenAI models are reasoning models, which may not accept one.
        # A stuck request fails after 60 seconds; short network errors are retried twice.
        return ChatOpenAI(model=model, timeout=60, max_retries=2)  # reads OPENAI_API_KEY from the environment
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=0)
