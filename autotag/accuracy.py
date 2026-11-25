"""Run the small OpenJev model on the plain-text fixtures and record what happened.

Document-type labels are compared only to the intent file shipped with the
fixtures. That file was written with the samples. It is not a human study
and not an LLM judgment.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

from autotag.classify import DEFAULT_THRESHOLD
from autotag.industry import INDUSTRY_HYPOTHESIS
from autotag.openjev_backend import FALLBACK_MODEL, PREFERRED_MODEL, OpenJevBackend

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "measurements" / "cpu-accuracy.json"


def _rss_mb() -> float:
    # Linux: ru_maxrss is kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load_backend() -> tuple[OpenJevBackend, str | None, float, float]:
    """Prefer openjev-flash. If it does not load, try the 71M checkpoint."""
    errors: list[str] = []
    for model_id in (PREFERRED_MODEL, FALLBACK_MODEL):
        backend = OpenJevBackend(model_id)
        started = time.perf_counter()
        try:
            backend.load()
            warmup_started = time.perf_counter()
            backend.entail("warmup", ["This is a warmup hypothesis."])
            warmup_s = time.perf_counter() - warmup_started
            load_s = time.perf_counter() - started - warmup_s
            note = "; ".join(errors) if errors else None
            return backend, note, load_s, warmup_s
        except Exception as exc:  # the next checkpoint is the recovery path
            errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
    joined = " | ".join(errors)
    raise RuntimeError(f"Neither OpenJev checkpoint loaded. {joined}")


def run_accuracy(
    output: Path = DEFAULT_OUTPUT,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, object]:
    intent_path = FIXTURES / "intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    paths = sorted(path for path in FIXTURES.glob("*.txt") if path.is_file())
    missing = [path.name for path in paths if path.name not in intent]
    if missing:
        raise RuntimeError(f"intent.json is missing fixture intent for: {', '.join(missing)}")

    backend, fallback_note, load_s, warmup_s = _load_backend()
    documents: list[dict[str, object]] = []
    matches = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        started = time.perf_counter()
        industry = backend.propose_industry(text, threshold)
        doc_type = backend.classify_doc_type(text, threshold)
        elapsed = time.perf_counter() - started
        expected = intent[path.name]
        matched = doc_type.label == expected
        matches += int(matched)
        documents.append(
            {
                "doc_id": path.name,
                "latency_s": round(elapsed, 4),
                "industry": industry.to_dict(),
                "doc_type": doc_type.to_dict(),
                "fixture_intent_doc_type": expected,
                "doc_type_matches_fixture_intent": matched,
            }
        )

    payload: dict[str, object] = {
        "model_id": backend.model_id,
        "revision": backend.revision,
        "device": backend.device,
        "dtype": backend.dtype,
        "threads": getattr(backend, "threads", None),
        "threshold": threshold,
        "industry_hypothesis": INDUSTRY_HYPOTHESIS,
        "score": "P(entailment) from the 3-way softmax, not renormalized across labels",
        "load_s": round(load_s, 4),
        "warmup_s": round(warmup_s, 4),
        "peak_rss_mb": round(_rss_mb(), 1),
        "fallback_note": fallback_note,
        "documents": documents,
        "fixture_intent_doc_type": {
            "n": len(documents),
            "matches": matches,
            "meaning": (
                "Count of document-type labels equal to autotag/fixtures/intent.json. "
                "The intent file was written with the samples. This is not agreement "
                "with a person or with a large language model."
            ),
        },
        "not_measured": [
            "Agreement with a human labeler. No labeled set of about 50 documents, and no second annotator.",
            "Agreement with a large language model. No comparison model was named or run.",
            "A threshold fit on held-out labels. 0.5 was fixed before reading these scores.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def format_report(payload: dict[str, object]) -> str:
    lines = [
        f"model: {payload['model_id']}",
        f"revision: {payload['revision'] or '(unknown)'}",
        f"device: {payload['device']} dtype: {payload['dtype']} threads: {payload['threads']}",
        f"threshold: {payload['threshold']}",
        f"load_s: {payload['load_s']} warmup_s: {payload['warmup_s']} peak_rss_mb: {payload['peak_rss_mb']}",
    ]
    if payload.get("fallback_note"):
        lines.append(f"fallback: {payload['fallback_note']}")
    documents = payload["documents"]
    assert isinstance(documents, list)
    for row in documents:
        assert isinstance(row, dict)
        industry = row["industry"]
        doc_type = row["doc_type"]
        assert isinstance(industry, dict) and isinstance(doc_type, dict)
        industry_flag = " abstain" if industry["abstained"] else ""
        type_flag = " abstain" if doc_type["abstained"] else ""
        mark = "match" if row["doc_type_matches_fixture_intent"] else "differ"
        lines.append(
            f"{row['doc_id']}: industry {industry['label']} ({industry['score']}){industry_flag}; "
            f"doc_type {doc_type['label']} ({doc_type['score']}){type_flag}; "
            f"fixture {row['fixture_intent_doc_type']} {mark}; {row['latency_s']}s"
        )
    summary = payload["fixture_intent_doc_type"]
    assert isinstance(summary, dict)
    lines.append(f"fixture intent doc_type: {summary['matches']}/{summary['n']}")
    return "\n".join(lines)
