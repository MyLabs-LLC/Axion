"""Agreement between two labelers (human, JEV, or an LLM).

The rate is undefined until both sides have labeled the same documents.
These helpers are the measurement, not a result.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def accuracy(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Label sequences differ in length: {len(left)} vs {len(right)}.")
    if not left:
        raise ValueError("Need at least one paired label.")
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return matches / len(left)


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    """Chance-corrected agreement. 1 is perfect, 0 is chance, negative is worse."""
    if len(left) != len(right):
        raise ValueError(f"Label sequences differ in length: {len(left)} vs {len(right)}.")
    if not left:
        raise ValueError("Need at least one paired label.")
    observed = accuracy(left, right)
    count = len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum((left_counts[label] / count) * (right_counts[label] / count) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def pairwise_report(pairs: dict[str, Sequence[str]]) -> dict[str, dict[str, float]]:
    """Every unordered pair of named labelers -> accuracy and Cohen's kappa."""
    names = list(pairs)
    report: dict[str, dict[str, float]] = {}
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            key = f"{left_name}_vs_{right_name}"
            report[key] = {
                "accuracy": accuracy(pairs[left_name], pairs[right_name]),
                "cohen_kappa": cohen_kappa(pairs[left_name], pairs[right_name]),
                "n": float(len(pairs[left_name])),
            }
    return report
