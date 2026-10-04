# Greater Manchester Ghostbusters Chatbot

This is Ghostbusters AI chatbot to answer all your deep lore, equipment, spritual questions about spooks, spectres or ghosts within the Ghostbusters universe and general franchise questions

It is powered by Meta's Llama 3.1 8B model via their Ollama framework and is designed to run locally saving compute and token usage and ensuring your data, prompts, and files never leave your device, preventing third-party data tracking or external training.

## Requirements and installation

- Python 3.11+
- Ollama
- Llama 3.1 8B

You first need to download the Ollama framework globally on to your machine so you can then pull and run models locally

https://ollama.com/download

Also ensure you have python installed using [Homebrew](https://brew.sh/) if using Mac/Linux or if on Windows the [Python Windows Download Page](https://www.python.org/downloads/windows/)

Create a virtual environment:

`python -m venv .venv`

Install dependencies:

`pip install -r requirements.txt`

Download Llama model

`ollama pull llama3.1:8b`

To run the chatbot and ask questions:

`python main.py`

To run the tests that use pytest and a fake LLM and a fake classifier to simulate the behavior of the chatbot without relying on Llama having to be spun up

`pytest -q`
