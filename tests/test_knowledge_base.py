from unittest.mock import Mock

import pytest

from app.exceptions import EmbeddingError, RetrievalError
from app.knowledge_base import KnowledgeBase


def test_chunk_text_splits_text():
    text = "A" * 2500

    chunks = KnowledgeBase.chunk_text(
        text,
        chunk_size=1000,
        overlap=200,
    )

    assert len(chunks) == 4
    assert all(chunks)


def test_chunk_text_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        KnowledgeBase.chunk_text(
            "test",
            chunk_size=0,
            overlap=0,
        )


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        KnowledgeBase.chunk_text(
            "test",
            chunk_size=100,
            overlap=100,
        )


def test_embed_returns_embeddings(monkeypatch):
    knowledge_base = KnowledgeBase()

    fake_response = {
        "embeddings": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    }

    mock_embed = Mock(return_value=fake_response)

    monkeypatch.setattr(
        knowledge_base.ollama,
        "embed",
        mock_embed,
    )

    result = knowledge_base.embed(
        ["first text", "second text"]
    )

    assert result == fake_response["embeddings"]

    mock_embed.assert_called_once()


def test_embed_raises_embedding_error(monkeypatch):
    knowledge_base = KnowledgeBase()

    def failing_embed(*args, **kwargs):
        raise RuntimeError("Ollama unavailable")

    monkeypatch.setattr(
        knowledge_base.ollama,
        "embed",
        failing_embed,
    )

    with pytest.raises(EmbeddingError):
        knowledge_base.embed(["test"])


def test_get_existing_ids():
    knowledge_base = KnowledgeBase()

    mock_collection = Mock()

    mock_collection.get.side_effect = [
        {"ids": ["id1"]},
        {"ids": ["id3"]},
    ]

    knowledge_base.collection = mock_collection

    result = knowledge_base.get_existing_ids(
        ["id1", "id2", "id3"],
        check_batch_size=2,
    )

    assert result == {"id1", "id3"}


def test_add_pages_skips_existing_chunks(monkeypatch):
    knowledge_base = KnowledgeBase()

    existing_id = None

    def fake_get_existing_ids(ids):
        nonlocal existing_id

        existing_id = ids[0]

        return {existing_id}

    monkeypatch.setattr(
        knowledge_base,
        "get_existing_ids",
        fake_get_existing_ids,
    )

    embed_mock = Mock()

    monkeypatch.setattr(
        knowledge_base,
        "embed",
        embed_mock,
    )

    page = {
        "source_type": "tobin",
        "source_name": "Tobin's Spirit Guide",
        "title": "Test Page",
        "page": 1,
        "url": "",
        "text": "This is a test Ghostbusters document.",
    }

    result = knowledge_base.add_pages([page])

    assert result == 0

    embed_mock.assert_not_called()


def test_add_pages_embeds_and_stores_new_chunks(monkeypatch):
    knowledge_base = KnowledgeBase()

    monkeypatch.setattr(
        knowledge_base,
        "get_existing_ids",
        lambda ids: set(),
    )

    monkeypatch.setattr(
        knowledge_base,
        "embed",
        lambda documents: [
            [0.1, 0.2, 0.3]
            for _ in documents
        ],
    )

    mock_collection = Mock()

    knowledge_base.collection = mock_collection

    page = {
        "source_type": "tobin",
        "source_name": "Tobin's Spirit Guide",
        "title": "Test Page",
        "page": 1,
        "url": "",
        "text": "This is a test Ghostbusters document.",
    }

    result = knowledge_base.add_pages([page])

    assert result == 1

    mock_collection.upsert.assert_called_once()

    call_kwargs = mock_collection.upsert.call_args.kwargs

    assert len(call_kwargs["ids"]) == 1
    assert len(call_kwargs["documents"]) == 1
    assert len(call_kwargs["metadatas"]) == 1
    assert len(call_kwargs["embeddings"]) == 1

def test_retrieve_returns_tobin_results_first(monkeypatch):
    knowledge_base = KnowledgeBase()

    monkeypatch.setattr(
        knowledge_base,
        "embed",
        lambda texts: [[0.1, 0.2, 0.3]],
    )

    query_calls = []

    def fake_query(**kwargs):
        source_type = kwargs["where"]["source_type"]
        query_calls.append(source_type)

        if source_type == "tobin":
            return {
                "documents": [
                    ["A Tobin result."]
                ],
                "metadatas": [
                    [{
                        "source_type": "tobin",
                        "source_name": "Tobin's Spirit Guide",
                        "title": "Ghost Entry",
                        "page": 10,
                        "url": "",
                    }]
                ],
                "distances": [
                    [0.15]
                ],
            }

        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    mock_collection = Mock()
    mock_collection.query.side_effect = fake_query

    knowledge_base.collection = mock_collection

    results = knowledge_base.retrieve(
        "Tell me about this ghost."
    )

    print("Query calls:", query_calls)
    print("Results:", results)
    assert len(results) == 1
    assert results[0].source_type == "tobin"
    assert results[0].text == "A Tobin result."

    assert query_calls == ["tobin"]

def test_retrieve_raises_retrieval_error(monkeypatch):
    knowledge_base = KnowledgeBase()

    monkeypatch.setattr(
        knowledge_base,
        "embed",
        lambda texts: [[0.1, 0.2, 0.3]],
    )

    mock_collection = Mock()

    mock_collection.query.side_effect = RuntimeError(
        "Chroma unavailable"
    )

    knowledge_base.collection = mock_collection

    with pytest.raises(RetrievalError):
        knowledge_base.retrieve("test question")