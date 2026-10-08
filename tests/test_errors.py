from app.exceptions import (
    DocumentLoadError,
    EmbeddingError,
    GhostbustersAppError,
    IngestionError,
    LLMError,
    RetrievalError,
)


def test_all_application_errors_have_common_base():
    assert issubclass(
        LLMError,
        GhostbustersAppError,
    )

    assert issubclass(
        EmbeddingError,
        GhostbustersAppError,
    )

    assert issubclass(
        RetrievalError,
        GhostbustersAppError,
    )

    assert issubclass(
        DocumentLoadError,
        GhostbustersAppError,
    )

    assert issubclass(
        IngestionError,
        GhostbustersAppError,
    )