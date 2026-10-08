
import argparse

from pathlib import Path

from numpy import rint
import pymupdf

from app.exceptions import (
    DocumentLoadError,
    EmbeddingError,
    IngestionError,
)

from app.config import settings
from app.logging_config import configure_logging

from app.document_loader import load_pdf
from app.knowledge_base import KnowledgeBase




def main()-> None:

    logger = configure_logging()

    parser = argparse.ArgumentParser(
        description="Ingest a Ghostbusters PDF into the knowledge base."
    )

    parser.add_argument(
        "pdf",
        help="Path to the PDF file."
    )

    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on every PDF page."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="OCR rendering resolution (default: 300)."
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=300,
        help="Number of extracted characters to preview per page."
    )

    try: 
        args = parser.parse_args()

        logger.info(
            "Starting ingestion: %s",
            args.pdf,
        )

        pdf_path = Path(args.pdf)

        if not pdf_path.exists():
         parser.error(f"PDF not found: {pdf_path}")

        if args.dpi < 72:
            parser.error("DPI must be at least 72.")

        print("\n========================================")
        print(" GHOSTBUSTERS KNOWLEDGE BASE INGESTION")
        print("========================================")

        print(f"\nPDF: {pdf_path}")
        print(f"File size: {pdf_path.stat().st_size / (1024 * 1024):.2f} MB")

        # Check PDF page count
        with pymupdf.open(pdf_path) as pdf:
            total_pages = len(pdf)

        print(f"Total PDF pages: {total_pages}")
        print(f"OCR enabled: {args.force_ocr}")
        print(f"OCR resolution: {args.dpi} DPI")

        print("\nStarting text extraction...\n")

        # Extract text from PDF
        pages = load_pdf(
            pdf_path,
            force_ocr=args.force_ocr,
            cache_dir="data/ocr_cache",
        )

        logger.info(
            "PDF extraction completed. Pages: %d",
            len(pages),
        )

        if not pages:
            logger.error("ERROR: No text was extracted from the PDF.")
            return

        # Report extraction results
        ocr_pages = sum(
            page["extraction_method"] == "ocr"
            for page in pages
        )

        native_pages = sum(
            page["extraction_method"] == "native"
            for page in pages
        )

        print("\n========================================")
        print(" TEXT EXTRACTION RESULTS")
        print("========================================")

        print(f"Pages with extracted text: {len(pages)}")
        print(f"Pages processed using OCR: {ocr_pages}")
        print(f"Pages processed using native extraction: {native_pages}")

        total_characters = sum(
            len(page["text"])
            for page in pages
        )

        print(f"Total characters extracted: {total_characters:,}")

        # Display a preview of every extracted page
        print("\n========================================")
        print(" EXTRACTED TEXT PREVIEW")
        print("========================================")

        for page in pages:
            print(
                f"\nPage {page['page']} "
                f"[{page['extraction_method'].upper()}]"
            )

            print(f"Characters extracted: {len(page['text']):,}")

            print("-" * 50)

            preview = page["text"][:args.preview]
            print(preview)

            if len(page["text"]) > args.preview:
                print("... [preview truncated]")

            print("-" * 50)

        # Confirm before ingesting
        print("\n========================================")
        print(" CHROMADB INGESTION")
        print("========================================")

        print("Preparing extracted text for vector storage...")

        knowledge_base = KnowledgeBase()

        logger.info(
            "Existing chunks in database: %d",
            knowledge_base.count(),
        )

        print(f"Existing chunks in database: {knowledge_base.count()}")

        added = knowledge_base.add_pages(pages)

        print("\n========================================")
        print(" INGESTION COMPLETE")
        print("========================================")

        print(f"Chunks submitted: {added}")
        print(f"Total chunks in database: {knowledge_base.count()}")

        print("\nPDF ingestion finished successfully.")

    except DocumentLoadError as exc:

        logger.error(
            "Document loading failed: %s",
            exc,
        )

        print()
        print("ERROR: Unable to process the PDF.")
        print(exc)

    except EmbeddingError as exc:

        logger.error(
            "Embedding failed: %s",
            exc,
        )

        print()
        print("ERROR: Unable to generate embeddings.")
        print(exc)
        print()
        print(
            "Check that Ollama is running and that "
            f"'{settings.embedding_model}' is available."
        )

    except IngestionError as exc:

        logger.error(
            "Ingestion failed: %s",
            exc,
        )

        print()
        print("ERROR: Ingestion failed.")
        print(exc)

    except KeyboardInterrupt:

        logger.warning(
            "Ingestion interrupted by user."
        )

        print()
        print("Ingestion interrupted.")
        print("Previously completed batches remain stored.")

    except Exception as exc:

        logger.exception(
            "Unexpected ingestion error."
        )

        print()
        print(
            "ERROR: An unexpected error occurred."
        )
        print(
            "See logs/ghostbusters.log for details."
        )


if __name__ == "__main__":
    main()













































