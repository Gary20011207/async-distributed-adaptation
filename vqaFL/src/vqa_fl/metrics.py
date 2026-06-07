from __future__ import annotations

from vqa_fl.data import OPTION_LETTERS


def parse_choice(text: str) -> str:
    normalized = text.strip().upper()
    for char in normalized:
        if char in OPTION_LETTERS:
            return char
    return ""


def option_accuracy(predictions: list[str], answers: list[str]) -> float:
    if len(predictions) != len(answers):
        raise ValueError("predictions and answers must have the same length")
    if not answers:
        return 0.0
    correct = 0
    for prediction, answer in zip(predictions, answers, strict=True):
        correct += int(parse_choice(prediction) == parse_choice(answer))
    return correct / len(answers)

