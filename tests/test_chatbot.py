from typing import Any, cast

from app.chatbot import GhostbustersChatbot

# Tests currently use a fake LLM and a fake classifier to simulate the behavior of the chatbot without relying on external dependencies. 
# This allows for isolated testing of the chatbot's logic and response handling.

class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = []

    def chat(self, messages, temperature=None) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
            }
        )

        return self.responses.pop(0)


class FakeClassifier:
    def __init__(self, in_scope: bool) -> None:
        self.in_scope = in_scope
        self.calls = []

    def classify(self, user_message: str):
        self.calls.append(user_message)

        class Result:
            in_scope: bool
            topic: str
            reason: str

        result = Result()
        result.in_scope = self.in_scope
        result.topic = "test"
        result.reason = "Test result"

        return result


def test_chatbot_rejects_out_of_scope_question() -> None:
    llm = FakeLLM(
        ["This response should never be used."]
    )

    classifier = FakeClassifier(
        in_scope=False,
    )

    chatbot = GhostbustersChatbot(
        llm=cast(Any, llm),
        classifier=cast(Any, classifier),
    )

    response = chatbot.answer(
        "What is the weather today?"
    )

    assert "only answer questions about" in response
    assert len(llm.calls) == 0


def test_chatbot_answers_in_scope_question() -> None:
    llm = FakeLLM(
        ["Egon Spengler was a founding Ghostbuster."]
    )

    classifier = FakeClassifier(
        in_scope=True,
    )

    chatbot = GhostbustersChatbot(
        llm=cast(Any, llm),
        classifier=cast(Any, classifier),
    )

    response = chatbot.answer(
        "Who is Egon Spengler?"
    )

    assert response == (
        "Egon Spengler was a founding Ghostbuster."
    )

    assert len(llm.calls) == 1


def test_chatbot_preserves_conversation_history() -> None:
    llm = FakeLLM(
        [
            "Peter Venkman is a Ghostbuster.",
            "He is portrayed by Bill Murray in the original films.",
        ]
    )

    classifier = FakeClassifier(
        in_scope=True,
    )

    chatbot = GhostbustersChatbot(
        llm=cast(Any, llm),
        classifier=cast(Any, classifier),
    )

    chatbot.answer(
        "Who is Peter Venkman?"
    )

    chatbot.answer(
        "Who portrays him?"
    )
    assert len(chatbot.history) == 4

    second_call_messages = llm.calls[1]["messages"]

    assert any(
        message["content"] == "Who is Peter Venkman?"
        for message in second_call_messages
    )


def test_out_of_scope_question_does_not_modify_history() -> None:
    llm = FakeLLM(
        ["This response should not be used."]
    )

    classifier = FakeClassifier(
        in_scope=False,
    )

    chatbot = GhostbustersChatbot(
        llm=cast(Any, llm),
        classifier=cast(Any, classifier),
    )

    chatbot.answer(
        "Tell me about football."
    )

    assert chatbot.history == []