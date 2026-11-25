"""CPU entailment with a small OpenJev checkpoint.

torch and transformers are imported inside `load` so the stub path stays
on the standard library. Install them from autotag/requirements-openjev.txt
into a separate environment. They are not Axion training dependencies.
"""

from __future__ import annotations

import os
from typing import Any

from autotag.classify import DEFAULT_THRESHOLD, _classification, doc_type_hypotheses, doc_type_names
from autotag.industry import industry_candidates, industry_hypothesis

PREFERRED_MODEL = "xxparthparekhxx/openjev-flash"
FALLBACK_MODEL = "RazvanManolache/raz-systemone-nli-xsmall-openjev"
MAX_LENGTH = 512
BATCH = 8


class OpenJevBackend:
    """Score hypotheses with P(entailment). Scores are not renormalized across labels."""

    def __init__(self, model_id: str = PREFERRED_MODEL, revision: str | None = None) -> None:
        self.model_id = model_id
        self.revision = revision or ""
        self.device = "cpu"
        self.dtype = "float32"
        self._model: Any = None
        self._tokenizer: Any = None
        self._entailment_index = 1

    @property
    def name(self) -> str:
        return self.model_id

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        threads = os.cpu_count() or 1
        torch.set_num_threads(threads)
        self.threads = threads
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        model.eval()
        model.to("cpu")
        self._entailment_index = _entailment_index(model.config)
        self._tokenizer = tokenizer
        self._model = model
        self.revision = _cached_revision(self.model_id)

    def entail(self, premise: str, hypotheses: list[str]) -> list[float]:
        """P(entailment) for each hypothesis, in order. Empty input returns []."""
        if not hypotheses:
            return []
        self.load()
        import torch

        scores: list[float] = []
        assert self._model is not None and self._tokenizer is not None
        for start in range(0, len(hypotheses), BATCH):
            chunk = hypotheses[start : start + BATCH]
            encoded = self._tokenizer(
                [premise] * len(chunk),
                chunk,
                truncation="only_first",
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self._model(**encoded).logits.float()
                probabilities = torch.softmax(logits, dim=-1)
            column = probabilities[:, self._entailment_index]
            scores.extend(float(value) for value in column.tolist())
        return scores

    def propose_industry(self, text: str, threshold: float = DEFAULT_THRESHOLD):
        phrases = industry_candidates(text)
        hypotheses = [industry_hypothesis(phrase) for phrase in phrases]
        probabilities = self.entail(text, hypotheses)
        scores = dict(zip(phrases, probabilities))
        return _classification("industry", scores, threshold, self.name, {})

    def classify_doc_type(self, text: str, threshold: float = DEFAULT_THRESHOLD):
        hypotheses = doc_type_hypotheses()
        ordered = list(hypotheses)
        probabilities = self.entail(text, [hypotheses[label_id] for label_id in ordered])
        scores = dict(zip(ordered, probabilities))
        return _classification("doc_type", scores, threshold, self.name, doc_type_names())


def _entailment_index(config: Any) -> int:
    raw = getattr(config, "id2label", {}) or {}
    labels = {int(key): str(value).lower() for key, value in raw.items()}
    for index, name in labels.items():
        if "entail" in name:
            return index
    raise RuntimeError(f"Checkpoint labels have no entailment class: {labels}")


def _cached_revision(model_id: str) -> str:
    """Commit hash recorded by the Hub cache, when the snapshot has one."""
    try:
        from huggingface_hub import try_to_load_from_cache

        resolved = try_to_load_from_cache(model_id, "config.json")
    except (ImportError, OSError):
        return ""
    if not isinstance(resolved, str):
        return ""
    parts = resolved.split(os.sep)
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1]
    return ""
