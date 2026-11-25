"""Smoke checks for the fixture pipeline. Not a claim about model accuracy."""

from __future__ import annotations

from pathlib import Path

from autotag.agreement import accuracy, cohen_kappa
from autotag.classify import StubClassifier, classify_document
from autotag.cluster import LexicalTfEmbedder, cluster_texts

FIXTURES = Path(__file__).resolve().parent / "fixtures"

EXPECTED = {
    "hospital_invoice.txt": ("62", "invoice"),
    "discharge_summary.txt": ("62", "report"),
    "saas_agreement.txt": ("51", "contract"),
}


def main() -> None:
    classifier = StubClassifier()
    items: list[tuple[str, str]] = []
    for name, (industry, doc_type) in EXPECTED.items():
        text = (FIXTURES / name).read_text(encoding="utf-8")
        result = classify_document(text, classifier)
        got = (result["industry"].label, result["doc_type"].label)
        if got != (industry, doc_type):
            raise SystemExit(f"{name}: expected {(industry, doc_type)}, got {got}")
        if result["industry"].abstained or result["doc_type"].abstained:
            raise SystemExit(f"{name}: unexpected abstention")
        items.append((name, text))

    clusters, _similarity = cluster_texts(items, LexicalTfEmbedder(), classifier, threshold=0.2)
    if len(clusters) != 2:
        raise SystemExit(f"expected 2 clusters, got {len(clusters)}")
    grouped = {frozenset(member.doc_id for member in cluster.members) for cluster in clusters}
    healthcare = frozenset({"hospital_invoice.txt", "discharge_summary.txt"})
    software = frozenset({"saas_agreement.txt"})
    if grouped != {healthcare, software}:
        raise SystemExit(f"unexpected groups: {grouped}")
    healthcare_cluster = next(cluster for cluster in clusters if len(cluster.members) == 2)
    if healthcare_cluster.suggested_industry != "62":
        raise SystemExit("healthcare cluster should agree on industry 62")
    if healthcare_cluster.suggested_doc_type is not None:
        raise SystemExit("mixed document types should not collapse to one cluster tag")

    perfect_left = ["62", "62", "51"]
    perfect_right = ["62", "62", "51"]
    if accuracy(perfect_left, perfect_right) != 1.0 or cohen_kappa(perfect_left, perfect_right) != 1.0:
        raise SystemExit("agreement helpers failed on identical labels")
    partial = cohen_kappa(["62", "62", "51"], ["62", "51", "51"])
    if not -1.0 <= partial < 1.0:
        raise SystemExit(f"kappa out of range: {partial}")
    print("autotag selfcheck ok")


if __name__ == "__main__":
    main()
