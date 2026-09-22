from typing import Any, cast

from app.classifier import TopicClassifier

# Tests currently use a fake LLM to simulate the behavior of the classifier without relying on external dependencies.
# This test suite ensures that the TopicClassifier behaves correctly when given various responses from the LLM, 
# including valid and invalid JSON, as well as handling Markdown code fences.

class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def chat(self, messages, _temperature=None) -> str:
        return self.response


def test_classifier_accepts_ghostbusters_question() -> None:
    llm = FakeLLM(
        """
        {
          "in_scope": true,
          "topic": "characters",
          "reason": "The question concerns a Ghostbusters character."
        }
        """
    )

    classifier = TopicClassifier(cast(Any, llm))

    result = classifier.classify(
        "Who is Peter Venkman?"
    )

    assert result.in_scope is True
    assert result.topic == "characters"


def test_classifier_rejects_unrelated_question() -> None:
    llm = FakeLLM(
        """
        {
          "in_scope": false,
          "topic": "cooking",
          "reason": "The question concerns cooking."
        }
        """
    )

    classifier = TopicClassifier(cast(Any, llm))

    result = classifier.classify(
        "How do I cook pasta?"
    )

    assert result.in_scope is False
    assert result.topic == "cooking"


def test_classifier_handles_invalid_json() -> None:
    llm = FakeLLM("This is not JSON")

    classifier = TopicClassifier(cast(Any, llm))

    result = classifier.classify(
        "Who is Ray Stantz?"
    )

    assert result.in_scope is False
    assert result.topic == "invalid_classifier_response"


def test_classifier_handles_markdown_json() -> None:
    llm = FakeLLM(
        """
        ```json
        {
          "in_scope": true,
          "topic": "equipment",
          "reason": "The question concerns Ghostbusters equipment."
        }
        ```
        """
    )

    classifier = TopicClassifier(cast(Any, llm))

    result = classifier.classify(
        "What does a proton pack do?"
    )

    assert result.in_scope is True
    assert result.topic == "equipment"


def test_classifier_rejects_invalid_boolean() -> None:
    llm = FakeLLM(
        """
        {
          "in_scope": "yes",
          "topic": "characters",
          "reason": "Invalid boolean test."
        }
        """
    )

    classifier = TopicClassifier(cast(Any, llm))

    result = classifier.classify(
        "Who is Winston Zeddemore?"
    )

    assert result.in_scope is False
    assert result.topic == "invalid_classifier_response"