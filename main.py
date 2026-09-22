from app.chatbot import GhostbustersChatbot
from app.classifier import TopicClassifier
from app.llm import LocalLLM

# App entry point. Initializes the chatbot and handles user input/output.

def print_banner() -> None:
    print("=" * 60)
    print("Dogs & Cats living together, Mass Hysteria!: a Ghostbusters Chatbot")
    print("Powered by Llama through Ollama")
    print("=" * 60)
    print("Ask questions about the Ghostbusters universe.")
    print("Type 'exit' or 'quit' to leave.")
    print("=" * 60)
    print()


def main() -> None:
    print_banner()

    llm = LocalLLM()

    classifier = TopicClassifier(
        llm=llm,
    )

    chatbot = GhostbustersChatbot(
        llm=llm,
        classifier=classifier,
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            response = chatbot.answer(user_input)
            print(f"\nAssistant: {response}\n")

        except Exception as exc:
            print(
                "\nAn error occurred while communicating with Ollama:"
                f"\n{exc}\n"
            )


if __name__ == "__main__":
    main()