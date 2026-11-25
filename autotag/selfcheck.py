"""Smoke checks for the stub pipeline. Not a claim about OpenJev accuracy."""

from __future__ import annotations

import json
from pathlib import Path

from autotag.agreement import accuracy, cohen_kappa
from autotag.classify import StubBackend, classify_document
from autotag.cluster import LexicalTfEmbedder, cluster_texts

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CLUSTER_IDS = ("hospital_invoice.txt", "discharge_summary.txt", "saas_agreement.txt")


def main() -> None:
    tagger = StubBackend()
    intent = json.loads((FIXTURES / "intent.json").read_text(encoding="utf-8"))
    texts: dict[str, str] = {}
    for path in sorted(FIXTURES.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        texts[path.name] = text
        expected = intent[path.name]
        result = classify_document(text, tagger)
        if result["doc_type"].label != expected:
            raise SystemExit(f"{path.name}: doc_type {result['doc_type'].label}, fixture intent {expected}")
        industry = result["industry"].label
        if industry != "other" and industry.casefold() not in text.casefold():
            raise SystemExit(f"{path.name}: industry {industry!r} is not a phrase from the document")
        if industry in {"11", "51", "62", "31-33"}:
            raise SystemExit(f"{path.name}: industry fell back to a NAICS code")

    items = [(name, texts[name]) for name in CLUSTER_IDS]
    clusters, _similarity = cluster_texts(items, LexicalTfEmbedder(), tagger, threshold=0.2)
    if len(clusters) != 2:
        raise SystemExit(f"expected 2 clusters, got {len(clusters)}")
    grouped = {frozenset(member.doc_id for member in cluster.members) for cluster in clusters}
    healthcare = frozenset({"hospital_invoice.txt", "discharge_summary.txt"})
    software = frozenset({"saas_agreement.txt"})
    if grouped != {healthcare, software}:
        raise SystemExit(f"unexpected groups: {grouped}")
    healthcare_cluster = next(cluster for cluster in clusters if len(cluster.members) == 2)
    if healthcare_cluster.suggested_doc_type is not None:
        raise SystemExit("mixed document types should not collapse to one cluster tag")

    if accuracy(["invoice", "report"], ["invoice", "report"]) != 1.0:
        raise SystemExit("accuracy helper failed")
    if cohen_kappa(["invoice", "report", "contract"], ["invoice", "report", "contract"]) != 1.0:
        raise SystemExit("kappa helper failed on identical labels")
    partial = cohen_kappa(["invoice", "report", "contract"], ["invoice", "contract", "contract"])
    if not -1.0 <= partial < 1.0:
        raise SystemExit(f"kappa out of range: {partial}")
    print("autotag selfcheck ok")


if __name__ == "__main__":
    main()
