from langchain_core.language_models import BaseChatModel


def build_chat_model(model: str) -> BaseChatModel:
    """An OpenAI model when the name starts with 'gpt', otherwise a local Ollama model."""
    if model.startswith("gpt"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0)  # reads OPENAI_API_KEY from the environment
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=0)
