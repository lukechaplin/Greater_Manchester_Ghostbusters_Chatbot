
from app.exceptions import DocumentLoadError 
import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path


import pymupdf
import pytesseract


from PIL import Image, ImageOps


def preprocess_image(image: Image.Image) -> Image.Image:
    """Prepare a scanned page for OCR."""
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    return image


def extract_page_text(
    page: pymupdf.Page,
    language: str = "eng",
    dpi: int = 300,
) -> str:
    """Render a PDF page and extract text using Tesseract."""

    scale = dpi / 72

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        alpha=False,
    )

    image = Image.open(
        BytesIO(pixmap.tobytes("png"))
    )

    image = preprocess_image(image)

    text = pytesseract.image_to_string(
        image,
        lang=language,
        config="--oem 3 --psm 6",
    )

    return text.strip()


def get_file_hash(path: Path) -> str:
    """Calculate a SHA-256 hash of the PDF file."""

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(block)

    return hasher.hexdigest()


def get_cache_directory(
    pdf_path: Path,
    cache_dir: str | Path,
    force_ocr: bool,
    language: str,
    minimum_text_length: int,
    dpi: int,
) -> Path:
    """
    Create a cache directory based on the PDF contents
    and the settings used to extract its text.
    """

    pdf_hash = get_file_hash(pdf_path)

    settings = {
        "force_ocr": force_ocr,
        "language": language,
        "minimum_text_length": minimum_text_length,
        "dpi": dpi,
    }

    settings_string = json.dumps(
        settings,
        sort_keys=True,
    )

    settings_hash = hashlib.sha256(
        settings_string.encode("utf-8")
    ).hexdigest()[:12]

    cache_name = f"{pdf_hash[:16]}_{settings_hash}"

    return Path(cache_dir) / cache_name
    


def load_cached_page(
    cache_file: Path,
) -> dict | None:
    """Load a previously cached page, if available."""

    if not cache_file.exists():
        return None

    try:
        with cache_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            cached_page = json.load(file)

        required_fields = {
            "text",
            "page",
            "title",
            "source_type",
            "source_name",
            "url",
            "extraction_method",
        }

        if not required_fields.issubset(cached_page):
            return None

        return cached_page

    except (OSError, json.JSONDecodeError):
        # A missing or damaged cache entry will be rebuilt.
        return None


def save_cached_page(
    cache_file: Path,
    page_data: dict,
) -> None:
    """Save a page to the cache using an atomic file replacement."""

    cache_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = cache_file.with_suffix(".tmp")

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            page_data,
            file,
            ensure_ascii=False,
        )

    temporary_file.replace(cache_file)


def load_pdf(
    path: str | Path,
    force_ocr: bool = False,
    language: str = "eng",
    minimum_text_length: int = 40,
    dpi: int = 300,
    cache_dir: str | Path = "data/ocr_cache",
) -> list[dict]:
    """
    Extract text from a PDF with page-level caching.

    Cached pages are reused when the PDF and extraction
    settings have not changed.

    Uncached pages are extracted and saved individually.
    This allows interrupted extraction runs to resume.

    Returns page text with source metadata and original
    page numbers.
    """

    logger = logging.getLogger(
    "ghostbusters.document_loader"
    )
    

    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    cache_path = get_cache_directory(
        pdf_path=pdf_path,
        cache_dir=cache_dir,
        force_ocr=force_ocr,
        language=language,
        minimum_text_length=minimum_text_length,
        dpi=dpi,
    )

    cache_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    pages = []

    cached_count = 0
    processed_count = 0
    ocr_count = 0
    native_count = 0

    logger.info(
        "Starting PDF extraction: %s",
        pdf_path,
    )

    logger.info(
    "OCR cache directory: %s",
    cache_path,
    )

    print()
    print("Preparing PDF extraction...")
    print(f"PDF: {pdf_path.name}")
    print(f"Cache directory: {cache_path}")

    try:

        with pymupdf.open(pdf_path) as pdf:

            total_pages = len(pdf)

        print(f"Total PDF pages: {total_pages}")
        print()

        for page_number, page in enumerate(
            pdf, # type: ignore
            start=1,
        ):

            cache_file = cache_path / (
                f"page_{page_number:04d}.json"
            )

            # Try to reuse the cached page.
            page_data = load_cached_page(cache_file)

            if page_data is not None:

                pages.append(page_data)
                cached_count += 1

                extraction_method = page_data[
                    "extraction_method"
                ]

                logger.debug(
                "Loaded page %d from OCR cache.",
                page_number,
                )

            else:

                # Extract the page because no valid cache exists.
                native_text = page.get_text("text").strip()

                if (
                    force_ocr
                    or len(native_text) < minimum_text_length
                ):

                    text = extract_page_text(
                        page,
                        language=language,
                        dpi=dpi,
                    )

                    extraction_method = "ocr"
                    ocr_count += 1

                else:

                    text = native_text
                    extraction_method = "native"
                    native_count += 1

                page_data = {
                    "text": text,
                    "page": page_number,
                    "title": pdf_path.stem,
                    "source_type": "tobin",
                    "source_name": "Tobin's Spirit Guide",
                    "url": "",
                    "extraction_method": extraction_method,
                }

                # Cache the page immediately so an interrupted
                # run does not lose completed extraction work.
                save_cached_page(
                    cache_file,
                    page_data,
                )

                processed_count += 1

                if text:
                    pages.append(page_data)

            # Display progress every five pages and at completion.
            if (
                page_number % 5 == 0
                or page_number == total_pages
            ):

                print(
                    f"Processed page {page_number}/{total_pages} "
                    f"| Cached: {cached_count} "
                    f"| Newly extracted: {processed_count}"
                )

        print()
        print("PDF EXTRACTION SUMMARY")
        print("----------------------")
        print(f"Total PDF pages: {total_pages}")
        print(f"Pages loaded from cache: {cached_count}")
        print(f"Pages newly extracted: {processed_count}")
        print(f"Pages processed using OCR: {ocr_count}")
        print(f"Pages processed using native extraction: {native_count}")
        print(f"Pages with extracted text: {len(pages)}")
        print()

        return pages

    except pymupdf.FileDataError as exc:

        logger.exception(
        "Unable to open PDF: %s",
        pdf_path,
        )
        
        raise DocumentLoadError(
        f"Unable to open PDF: {pdf_path}"
        ) from exc
    
    except Exception as exc:

        logger.exception(
        "Unexpected error while processing PDF: %s",
        pdf_path,
        )

        raise DocumentLoadError(
        f"Failed to process PDF: {pdf_path}"
        ) from exc