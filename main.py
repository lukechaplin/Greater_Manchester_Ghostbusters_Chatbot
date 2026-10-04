
from ollama import ResponseError

from app.llm import LocalLLM
from app.classifier import TopicClassifier
from app.chatbot import GhostbustersChatbot
from app.knowledge_base import KnowledgeBase
from app.config import settings

# App entry point. Initializes the chatbot and handles user input/output.

def main() -> None:

    print("=" * 60)

    print("Dogs & Cats living together, Mass Hysteria!: a Ghostbusters Chatbot")

    print("Powered by Llama through Ollama")

    print("=" * 60)

    print("Ask questions about the Ghostbusters universe.")

    print("Type 'exit' or 'quit' to leave.")

    print("=" * 60)

    print()

    try:
        # Initialise the local language model.
        llm = LocalLLM()

        # Initialise the topic classifier.
        classifier = TopicClassifier(llm)

        # Initialise the local knowledge base.
        knowledge_base = KnowledgeBase()

        print(f"\nAnswer model: {llm.model_name}")
        print(f"Embedding model: {settings.embedding_model}")

        # Initialise the chatbot with RAG enabled.
        chatbot = GhostbustersChatbot(
            llm=llm,
            classifier=classifier,
            
        )

        print(
            "\nKnowledge-base chunks:",
            knowledge_base.count(),
        )

    except Exception as error:
        print(
            "\nFailed to initialise the assistant."
        )
        print(f"Details: {error}")
        return

    print("\nAssistant ready!")

    print(
        "\nAsk me anything about the Ghostbusters franchise."
    )

    print("\nCommands:")
    print("  exit  - Quit the assistant")
    print("  quit  - Quit the assistant")
    print("  clear - Clear conversation history")

    print("\n" + "-" * 60)

    while True:

        try:
            user_message = input("\nYou: ").strip()

            if not user_message:
                continue

            if user_message.lower() in ("exit", "quit"):
                print(
                    "\nGhostbusters assistant shutting down. "
                    "Who you gonna call?"
                )
                break

            if user_message.lower() == "clear":
                chatbot.clear_history()
                print("\nConversation history cleared.")
                continue

            print("\nGhostbusters Assistant: ", end="", flush=True)

            response = chatbot.respond(user_message)

            print(response)

        except KeyboardInterrupt:
            print("\n\nAssistant interrupted.")
            break

        except EOFError:
            print("\n\nAssistant shutting down.")
            break

        except ResponseError as error:
            print(f"\nOllama response error: {error}")

        except Exception as error:
            print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()
































