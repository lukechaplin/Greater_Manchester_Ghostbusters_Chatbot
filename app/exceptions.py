class GhostbustersAppError(Exception):
    """Base exception for expected application errors."""


class LLMError(GhostbustersAppError):
    """Raised when an LLM operation fails."""


class EmbeddingError(GhostbustersAppError):
    """Raised when an embedding operation fails."""


class RetrievalError(GhostbustersAppError):
    """Raised when knowledge-base retrieval fails."""


class DocumentLoadError(GhostbustersAppError):
    """Raised when a document cannot be loaded or processed."""


class IngestionError(GhostbustersAppError):
    """Raised when document ingestion fails."""