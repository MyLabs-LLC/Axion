"""Zero-shot industry and document-type classification.

`Classifier.score` is the Jev `noul` contract: each label is an independent
P(yes), not a softmax. `StubClassifier` is a deterministic lexicon so the
pipeline runs with no model download. `JevClient` posts the same questions to
a local jevos server (`POST /v1/systemone`) and does not import a new package.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping, Protocol

from autotag.labels import ABSTAIN, labels_for, questions_for

# Phrase lists for the stub only. They are not shown to a real model.
# A label missing from this map scores the floor, then abstains.
_STUB_PHRASES: dict[str, tuple[str, ...]] = {
    "11": ("crop", "livestock", "harvest", "farm"),
    "21": ("oil well", "natural gas", "mining", "quarry"),
    "22": ("electric utility", "water utility", "kilowatt"),
    "23": ("construction", "contractor", "building permit"),
    "31-33": ("manufacturing", "factory", "assembly line"),
    "42": ("wholesale", "distributor"),
    "44-45": ("retail", "point of sale", "store receipt"),
    "48-49": ("freight", "warehouse", "shipment", "bill of lading"),
    "51": ("software", "saas", "hosted platform", "subscription"),
    "52": ("insurance policy", "interest rate", "underwriting", "loan agreement"),
    "53": ("real estate", "lease agreement", "tenant"),
    "54": ("law firm", "attorney", "accounting firm"),
    "55": ("holding company", "subsidiary management"),
    "56": ("staffing agency", "waste management"),
    "61": ("university", "curriculum", "tuition", "semester"),
    "62": ("hospital", "patient", "medical", "diagnosis", "inpatient"),
    "71": ("box office", "museum", "recreation"),
    "72": ("restaurant", "hotel reservation", "menu"),
    "81": ("dry cleaning", "auto repair", "membership dues"),
    "92": ("public administration", "municipal ordinance", "federal agency"),
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
_DEFAULT_THRESHOLD = 0.45
_DEFAULT_ENDPOINT = "http://127.0.0.1:8017/v1/systemone"


class Classifier(Protocol):
    name: str

    def score(self, text: str, questions: Mapping[str, str]) -> dict[str, float]:
        """Return P(yes) for every question id."""


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
    for char in text:
        if char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def _contains_phrase(text: str, phrase: str) -> bool:
    """True when `phrase` occurs as whole tokens, in order."""
    haystack = _split_words(text.lower())
    needle = _split_words(phrase.lower())
    if not needle:
        return False
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


def _phrase_score(text: str, phrases: tuple[str, ...]) -> float:
    hits = sum(1 for phrase in phrases if _contains_phrase(text, phrase))
    if hits == 0:
        return _FLOOR
    return min(0.93, 0.35 + 0.20 * hits)


class StubClassifier:
    """Deterministic stand-in for a CPU zero-shot model.

    Scores are a function of phrase hits only. The same text and questions
    always produce the same probabilities. This is not a calibrated model.
    """

    name = "stub"

    def score(self, text: str, questions: Mapping[str, str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for label_id in questions:
            phrases = _STUB_PHRASES.get(label_id, ())
            scores[label_id] = _phrase_score(text, phrases) if phrases else _FLOOR
        return scores


class JevClient:
    """Talk to a local jevos (or TypeSafe-compatible) server.

    No extra Python dependency. Start the server with the jevos binary, then
    pass ``--backend jev``. See autotag/README.md.
    """

    name = "jev"

    def __init__(self, endpoint: str = _DEFAULT_ENDPOINT, timeout_s: float = 30.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def score(self, text: str, questions: Mapping[str, str]) -> dict[str, float]:
        body = {
            "model": "jev-latest",
            "state": text,
            "questions": {
                label_id: {"type": "noul", "instructions": instruction}
                for label_id, instruction in questions.items()
            },
        }
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Could not reach a JEV server at {self.endpoint}. "
                "The default backend is the deterministic stub. "
                "To use the real CPU model, run jevos (`./jev serve`) and retry with --backend jev. "
                f"Underlying error: {exc}"
            ) from exc
        parsed = json.loads(raw)
        answers = parsed.get("answers")
        if not isinstance(answers, dict):
            raise RuntimeError("JEV response did not contain an answers object.")
        scores: dict[str, float] = {}
        for label_id in questions:
            entry = answers.get(label_id)
            if not isinstance(entry, dict) or "noul" not in entry:
                raise RuntimeError(f"JEV response is missing noul for {label_id!r}.")
            scores[label_id] = float(entry["noul"])
        return scores


def get_classifier(backend: str, endpoint: str = _DEFAULT_ENDPOINT) -> Classifier:
    if backend == "stub":
        return StubClassifier()
    if backend == "jev":
        return JevClient(endpoint=endpoint)
    raise ValueError(f"Unknown backend {backend!r}. Expected 'stub' or 'jev'.")


def _pick(scores: Mapping[str, float], threshold: float) -> tuple[str, float, bool]:
    if not scores:
        return ABSTAIN, 0.0, True
    # Stable tie break: higher score, then label id.
    label = max(scores, key=lambda key: (scores[key], key))
    score = scores[label]
    if score < threshold:
        return ABSTAIN, score, True
    return label, score, False


def _from_scores(
    backend: str,
    axis: str,
    scores: Mapping[str, float],
    threshold: float,
) -> Classification:
    axis_ids = [label.id for label in labels_for(axis)]
    missing = [label_id for label_id in axis_ids if label_id not in scores]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing scores for {axis}: {joined}")
    axis_scores = {label_id: float(scores[label_id]) for label_id in axis_ids}
    label, score, abstained = _pick(axis_scores, threshold)
    name = "Other" if abstained else next(item.name for item in labels_for(axis) if item.id == label)
    return Classification(
        axis=axis,
        label=label,
        label_name=name,
        score=score,
        abstained=abstained,
        scores=axis_scores,
        backend=backend,
    )


def classify_axis(
    classifier: Classifier,
    text: str,
    axis: str,
    threshold: float = _DEFAULT_THRESHOLD,
) -> Classification:
    scores = classifier.score(text, questions_for(axis))
    return _from_scores(classifier.name, axis, scores, threshold)


def classify_document(
    text: str,
    classifier: Classifier | None = None,
    threshold: float = _DEFAULT_THRESHOLD,
) -> dict[str, Classification]:
    """Classify one document on both axes in a single `score` call.

    One call matches jevos: every question about a text is one request, and
    the text is read once. Industry ids and document-type ids do not overlap.
    """
    model = classifier if classifier is not None else StubClassifier()
    questions = questions_for("industry") | questions_for("doc_type")
    scores = model.score(text, questions)
    return {
        "industry": _from_scores(model.name, "industry", scores, threshold),
        "doc_type": _from_scores(model.name, "doc_type", scores, threshold),
    }
