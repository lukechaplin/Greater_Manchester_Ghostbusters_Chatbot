
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import chromadb
import logging

from ollama import Client

from app.config import settings

from app.exceptions import EmbeddingError, RetrievalError


@dataclass
class RetrievedChunk:
    text: str
    source_type: str
    source_name: str
    title: str
    page: int
    url: str
    distance: float


class KnowledgeBase:
    SOURCE_PRIORITY = ("tobin", "wiki", "web")

    def __init__(self) -> None:

        self.logger = logging.getLogger(
        "ghostbusters.knowledge_base"
        )

        self.ollama = Client(host=settings.ollama_host)

        Path(settings.chroma_path).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.chroma = chromadb.PersistentClient(
            path=settings.chroma_path
        )

        self.collection = self.chroma.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> list[str]:
        """Split text into overlapping chunks."""

        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid chunk size or overlap.")

        text = " ".join(text.split())

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Ollama."""

        if not texts:
            return []

        try:
            self.logger.debug(
            "Generating embeddings for %d texts.",
            len(texts),
            )

            response = self.ollama.embed(
            model=settings.embedding_model,
            input=texts,
            )

            embeddings = response["embeddings"]

            self.logger.debug(
            "Generated %d embeddings.",
            len(embeddings),
            )

            return response["embeddings"]

        except Exception as exc:
            self.logger.exception(
            "Embedding request failed."
            )
            raise EmbeddingError(
            "Unable to generate embeddings using "
            f"'{settings.embedding_model}'."
        ) from exc

    def get_existing_ids(
        self,
        ids: list[str],
        check_batch_size: int = 500,
    ) -> set[str]:
        """
        Check which chunk IDs already exist in ChromaDB.

        IDs are checked in batches to avoid sending an
        unnecessarily large request to ChromaDB.
        """

        existing_ids = set()

        for start in range(0, len(ids), check_batch_size):
            batch_ids = ids[start:start + check_batch_size]

            results = self.collection.get(
                ids=batch_ids,
            )

            existing_ids.update(results["ids"])

        return existing_ids

    def add_pages(self, pages: list[dict]) -> int:
        """
        Split extracted pages into chunks, skip chunks
        already stored in ChromaDB, embed missing chunks,
        and store them in batches.

        Returns the number of newly stored chunks.
        """

        documents = []
        metadatas = []
        ids = []

        # Prepare chunks and metadata.
        for page in pages:

            chunks = self.chunk_text(page["text"])

            for chunk_number, chunk in enumerate(chunks):

                metadata = {
                    "source_type": page["source_type"],
                    "source_name": page["source_name"],
                    "title": page["title"],
                    "page": int(page["page"]),
                    "url": page.get("url", ""),
                    "chunk_number": chunk_number,
                }

                # Generate a deterministic ID.
                identity = (
                    f'{metadata["source_type"]}|'
                    f'{metadata["title"]}|'
                    f'{metadata["page"]}|'
                    f'{chunk_number}|{chunk}'
                )

                chunk_id = sha256(
                    identity.encode("utf-8")
                ).hexdigest()

                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(chunk_id)

        if not documents:
            print("No chunks found to ingest.")
            return 0

        print(
            f"Prepared {len(documents)} chunks for ingestion."
        )

        # Identify chunks that already exist.
        existing_ids = self.get_existing_ids(ids)

        # Keep only chunks that are not already stored.
        new_documents = []
        new_metadatas = []
        new_ids = []

        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
        ):

            if chunk_id not in existing_ids:
                new_ids.append(chunk_id)
                new_documents.append(document)
                new_metadatas.append(metadata)

        skipped = len(existing_ids)

        print(f"Already stored: {skipped} chunks.")
        print(f"New chunks to ingest: {len(new_documents)}.")

        if not new_documents:
            print(
                "Knowledge base is already up to date."
            )
            return 0

        # Embed and store missing chunks in batches.
        batch_size = 16
        total_added = 0

        total_batches = (
            len(new_documents) + batch_size - 1
        ) // batch_size

        for start in range(0, len(new_documents), batch_size):

            end = min(
                start + batch_size,
                len(new_documents),
            )

            batch_number = (start // batch_size) + 1

            print(
                f"Embedding batch {batch_number}/{total_batches} "
                f"({start + 1}-{end} of {len(new_documents)} chunks)"
            )

            batch_documents = new_documents[start:end]
            batch_ids = new_ids[start:end]
            batch_metadatas = new_metadatas[start:end]

            # Generate embeddings.
            embeddings = self.embed(batch_documents)

            # Store this batch immediately.
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
                embeddings=embeddings, # type: ignore
            )

            total_added += len(batch_documents)

        print(
            f"Successfully stored {total_added} new chunks."
        )

        print(
            f"Total chunks currently in knowledge base: "
            f"{self.count()}"
        )

        return total_added

    def retrieve(
        self,
        question: str,
        limit_per_source: int = 4,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks, prioritising sources.

        Tobin's Spirit Guide is checked first, followed by
        the Ghostbusters Wiki and then web sources.
        """
        try:
            query_embedding = self.embed([question])[0]

            for source_type in self.SOURCE_PRIORITY:

                results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit_per_source,
                where={"source_type": source_type},
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

                documents = results["documents"][0] # type: ignore
                metadatas = results["metadatas"][0] # type: ignore
                distances = results["distances"][0] # type: ignore

                if documents:
                    return [
                        RetrievedChunk(
                        text=document,
                        source_type=metadata["source_type"], # type: ignore
                        source_name=metadata["source_name"], # type: ignore
                        title=metadata["title"], # type: ignore
                        page=metadata["page"], # type: ignore
                        url=metadata["url"], # type: ignore
                        distance=distance,
                    )
                    for document, metadata, distance in zip(
                        documents,
                        metadatas,
                        distances,
                    )
                ]
                
            return []
        
        except EmbeddingError:
            raise

        except Exception as exc:
            self.logger.exception(
            "Knowledge-base retrieval failed."
        )
            raise RetrievalError(
            "Unable to retrieve information "
            "from the knowledge base."
        ) from exc

    def count(self) -> int:
        """Return the total number of stored chunks."""

        return self.collection.count()