from dataclasses import dataclass
import os

# Configures app to use Llama by default

@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434",
    )

    model_name: str = os.getenv(
        "OLLAMA_MODEL",
        "llama3.1:8b",
    )

    temperature: float = 0.2

    max_history_messages: int = 10


settings = Settings()











