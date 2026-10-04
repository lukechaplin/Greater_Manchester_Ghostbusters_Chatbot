# Greater Manchester Ghostbusters Chatbot

This is Ghostbusters AI chatbot to answer all your deep lore, equipment, spritual questions about spooks, spectres or ghosts within the Ghostbusters universe and general franchise questions

It is powered by Meta's Llama 3.1 8B model via their Ollama framework and is designed to run locally saving compute and token usage and ensuring your data, prompts, and files never leave your device, preventing third-party data tracking or external training.

It uses its own training data and information sourced from the copy of Tobins Spirit Guide that is saved as a data source for RAG retrival in the ChromaDB vector database.

The guide (PDF) is composed of scanned images so an OCR (Optical Character Recognition) engine is used for text extraction during the initial PDF ingestion process.

The PDF is also cached at the same time and ingested using batch processing of the chunks generated to reduce failure rate during the embedding process into vectors and writing to the vector database (ChromaDB) instance.

I have implmeneted caching of the guide so the ingestion process if it needs to be re-run will not need to trigger the OCR engine to read the PDF unless there has been a change to the file.

Also batch processing is used to ingest the Spirit Guide to check which chunks are already stored in ChromaDB and skip chunks that have already been successfully ingested and only generate embeddings only for missing chunks.

## Requirements and installation

- Python 3.11+
- Ollama
- Llama 3.1 8B

You first need to download the Ollama framework globally on to your machine so you can then pull and run models locally

https://ollama.com/download

Also ensure you have python installed using [Homebrew](https://brew.sh/) if using Mac/Linux or if on Windows the [Python Windows Download Page](https://www.python.org/downloads/windows/)

You will also need to install an OCR (Optical Character Recognition) engine for text extraction from the scanned images from the Spirit Guide, this app is using Tesseract so install using Homebrew or if using windows via the [UB Mannhiem Tesseract Github](https://github.com/UB-Mannheim/tesseract/wiki)

Create a virtual environment:

`python -m venv .venv`

Install dependencies:

`pip install -r requirements.txt`

Download Llama model

`ollama pull llama3.1:8b`

To trigger the Tobins Spirit Guide (PDF) ingestion process run this command

`python ingest.py data/documents/tobin/Tobins_spirit_guide.pdf --force-ocr`

To run the chatbot and ask questions:

`python main.py`

To run the tests that use pytest and a fake LLM and a fake classifier to simulate the behavior of the chatbot without relying on Llama having to be spun up

`pytest -q`

Command to run to ingest Tobins spirit guide and add it to the RAG vector database

python ingest.py data/documents/tobin/Tobins_spirit_guide.pdf --force-ocr

I recommend ChromaDB for the local vector database and Ollama's nomic-embed-text model for embeddings. Keep llama3.1:8b as the model that writes the answers; the embedding model has a separate job.

We'll also make source priority an application-level rule, rather than relying on the LLM to remember it.

For the first working version, I’d build local PDF ingestion and retrieval first, then add the Wiki importer. That gives us a complete RAG loop we can test without depending on an external website.

One important detail: we should store the source name, document title, page number, and URL alongside every chunk. That lets the assistant identify where an answer came from and distinguish, for example, an RPG description from a film-continuity fact.
