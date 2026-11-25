"""Classify one document: one industry from the closed list, then one document type.

The stub is deterministic and uses the standard library. The OpenJev backend
lives in openjev_backend.py and is imported only when that backend is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from autotag.labels import ABSTAIN, DOC_TYPES, INDUSTRIES

# Phrase lists for the stub only. They are not shown to OpenJev.
_STUB_PHRASES: dict[str, tuple[str, ...]] = {
    "healthcare": ("hospital", "patient", "medical", "diagnosis", "inpatient", "clinic", "clinics"),
    "software": ("software", "saas", "hosted platform", "subscription"),
    "finance": ("bank", "interest rate", "investment portfolio", "mortgage"),
    "insurance": ("insurance", "premium", "insurance claim"),
    "legal": ("law firm", "attorney", "legal services"),
    "education": ("university", "tuition", "semester", "curriculum"),
    "energy": ("electric utility", "kilowatt", "natural gas", "oil well"),
    "manufacturing": ("factory", "assembly line", "manufacturing"),
    "retail": ("retail store", "point of sale"),
    "transportation": ("freight", "shipment", "passenger"),
    "real_estate": ("real estate", "tenant", "lease"),
    "construction": ("construction", "contractor", "building permit"),
    "government": ("public administration", "federal agency", "municipal"),
    "hospitality": ("restaurant", "hotel reservation"),
    "telecommunications": ("broadband", "telephone service", "broadcasting"),
    "agriculture": ("livestock", "harvest", "crop"),
    "invoice": ("invoice", "amount due", "bill to", "payment terms"),
    "receipt": ("receipt", "payment received", "paid in full"),
    "contract": ("agreement", "material breach", "terminate", "parties"),
    "policy": ("acceptable use", "this policy", "compliance policy"),
    "report": ("discharge summary", "findings", "diagnosis", "hospital course"),
    "letter": ("dear sir", "yours sincerely", "to whom it may concern"),
    "email": ("subject line", "email message"),
    "memo": ("memorandum", "internal memo"),
    "specification": ("specification", "functional requirement"),
    "financial_statement": ("balance sheet", "income statement", "cash flow statement"),
}

_FLOOR = 0.02
DEFAULT_THRESHOLD = 0.5


class Tagger(Protocol):
    name: str

    def propose_industry(self, text: str, threshold: float) -> Classification:
        """One industry from the closed list, or abstain."""

    def classify_doc_type(self, text: str, threshold: float) -> Classification:
        """One of the ten document types, or abstain."""


@dataclass(frozen=True)
class Classification:
    axis: str
    label: str
    label_name: str
    score: float
    abstained: bool
    scores: dict[str, float]
    backend: str

    def to_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "label": self.label,
            "label_name": self.label_name,
            "score": round(self.score, 4),
            "abstained": self.abstained,
            "backend": self.backend,
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
        }


def _split_words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def _contains_phrase(text: str, phrase: str) -> bool:
    haystack = _split_words(text)
    needle = _split_words(phrase)
    if not needle:
        return False
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


def _phrase_score(text: str, phrases: tuple[str, ...]) -> float:
    hits = sum(1 for phrase in phrases if _contains_phrase(text, phrase))
    if hits == 0:
        return _FLOOR
    return min(0.93, 0.35 + 0.20 * hits)


def pick_label(scores: Mapping[str, float], threshold: float) -> tuple[str, float, bool]:
    """Highest score wins. Ties break by label text. Below threshold abstains."""
    if not scores:
        return ABSTAIN, 0.0, True
    label = max(scores, key=lambda key: (scores[key], key))
    score = float(scores[label])
    if score < threshold:
        return ABSTAIN, score, True
    return label, score, False


def _classification(
    axis: str,
    scores: Mapping[str, float],
    threshold: float,
    backend: str,
    names: Mapping[str, str],
) -> Classification:
    label, score, abstained = pick_label(scores, threshold)
    if abstained:
        name = "Other"
    else:
        name = names.get(label, label)
    return Classification(
        axis=axis,
        label=label,
        label_name=name,
        score=score,
        abstained=abstained,
        scores=dict(scores),
        backend=backend,
    )


class StubBackend:
    """Lexicon stand-in. Scores are phrase hits on the closed label ids."""

    name = "stub"

    def propose_industry(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Classification:
        scores = {label.id: _phrase_score(text, _STUB_PHRASES[label.id]) for label in INDUSTRIES}
        names = {label.id: label.name for label in INDUSTRIES}
        return _classification("industry", scores, threshold, self.name, names)

    def classify_doc_type(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Classification:
        scores = {label.id: _phrase_score(text, _STUB_PHRASES[label.id]) for label in DOC_TYPES}
        names = {label.id: label.name for label in DOC_TYPES}
        return _classification("doc_type", scores, threshold, self.name, names)


def classify_document(
    text: str,
    tagger: Tagger | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Classification]:
    """Industry first, then document type. One label per axis, or abstain."""
    model = tagger if tagger is not None else StubBackend()
    return {
        "industry": model.propose_industry(text, threshold),
        "doc_type": model.classify_doc_type(text, threshold),
    }


