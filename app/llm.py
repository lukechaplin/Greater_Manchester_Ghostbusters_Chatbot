from collections.abc import Iterable

from ollama import Client

from app.config import settings

    # Wrapper around a locally running Ollama model.

class LocalLLM:
    
    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.model_name

        self.client = Client(
            host=host or settings.ollama_host,
        )

    def chat(
        self,
        messages: Iterable[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
      
        # Send messages to Ollama and return the response text.
    

        response = self.client.chat(
            model=self.model_name,
            messages=list(messages),
            options={
                "temperature": (
                    settings.temperature
                    if temperature is None
                    else temperature
                )
            },
        )

        return response["message"]["content"].strip()