"""Evaluation metrics for generative and classification tasks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

__all__ = ["classification_metrics", "exact_match_metrics", "normalize_text"]


def normalize_text(value: str) -> str:
    """Normalize generated text for deterministic reference comparison."""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def exact_match_metrics(expected: Iterable[str], predicted: Iterable[str]) -> dict:
    pairs = list(zip(expected, predicted, strict=True))
    if not pairs:
        raise ValueError("Cannot calculate metrics for an empty evaluation set")
    correct = sum(normalize_text(left) == normalize_text(right) for left, right in pairs)
    return {"total": len(pairs), "exact_match": correct / len(pairs)}


def classification_metrics(
    expected: Iterable[str], predicted: Iterable[str], labels: Iterable[str]
) -> dict:
    expected_values = list(expected)
    predicted_values = list(predicted)
    if len(expected_values) != len(predicted_values) or not expected_values:
        raise ValueError("Expected and predicted values must be non-empty and equally sized")

    canonical = {normalize_text(label): label for label in labels}
    expected_labels = [canonical.get(normalize_text(value), value) for value in expected_values]
    predicted_labels = [
        canonical.get(normalize_text(value), "INVALID") for value in predicted_values
    ]
    label_values = list(canonical.values())
    confusion: dict[str, Counter[str]] = {label: Counter() for label in label_values}
    for truth, guess in zip(expected_labels, predicted_labels, strict=True):
        confusion[truth][guess] += 1

    per_class: dict[str, dict[str, float]] = {}
    for label in label_values:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in label_values if other != label)
        false_negative = sum(count for guess, count in confusion[label].items() if guess != label)
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    total = len(expected_labels)
    correct = sum(
        left == right for left, right in zip(expected_labels, predicted_labels, strict=True)
    )
    invalid = predicted_labels.count("INVALID")
    return {
        "total": total,
        "accuracy": correct / total,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(per_class),
        "invalid_rate": invalid / total,
        "per_class": per_class,
        "confusion_matrix": {label: dict(confusion[label]) for label in label_values},
    }
