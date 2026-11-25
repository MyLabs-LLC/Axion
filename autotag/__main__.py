"""Command line for the AutoTag prototype.

From the repository root:

    python -m autotag classify autotag/fixtures/hospital_invoice.txt
    python -m autotag cluster autotag/fixtures
    python -m autotag.selfcheck
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autotag.classify import classify_document, get_classifier
from autotag.cluster import LexicalTfEmbedder, cluster_paths


def _print_classification(path: Path, text: str, backend: str, endpoint: str, threshold: float, as_json: bool) -> int:
    classifier = get_classifier(backend, endpoint)
    result = classify_document(text, classifier, threshold)
    if as_json:
        payload = {
            "path": str(path),
            "backend": classifier.name,
            "industry": result["industry"].to_dict(),
            "doc_type": result["doc_type"].to_dict(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"backend: {classifier.name}")
    print(f"file: {path}")
    for axis in ("industry", "doc_type"):
        item = result[axis]
        flag = " abstained" if item.abstained else ""
        print(f"{axis}: {item.label} — {item.label_name} ({item.score:.2f}){flag}")
    print("top scores:")
    for axis in ("industry", "doc_type"):
        ranked = sorted(result[axis].scores.items(), key=lambda pair: (-pair[1], pair[0]))[:3]
        rendered = ", ".join(f"{label}={score:.2f}" for label, score in ranked)
        print(f"  {axis}: {rendered}")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 2
    text = path.read_text(encoding="utf-8")
    try:
        return _print_classification(path, text, args.backend, args.endpoint, args.threshold, args.json)
    except (ConnectionError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 3


def _cmd_cluster(args: argparse.Namespace) -> int:
    folder = Path(args.directory)
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 2
    paths = sorted(path for path in folder.glob("*.txt") if path.is_file())
    if len(paths) < 2:
        print("Need at least two .txt documents to cluster.", file=sys.stderr)
        return 2
    try:
        classifier = get_classifier(args.backend, args.endpoint)
        embedder = LexicalTfEmbedder()
        clusters, doc_ids, similarity = cluster_paths(
            paths,
            embedder=embedder,
            classifier=classifier,
            threshold=args.similarity,
        )
    except (ConnectionError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if args.json:
        payload = {
            "backend": classifier.name,
            "embedder": embedder.name,
            "similarity_threshold": args.similarity,
            "doc_ids": doc_ids,
            "similarity": [[round(value, 4) for value in row] for row in similarity],
            "clusters": [cluster.to_dict() for cluster in clusters],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"backend: {classifier.name}")
    print(f"embedder: {embedder.name}")
    print("similarity:")
    for i, left in enumerate(doc_ids):
        for j in range(i + 1, len(doc_ids)):
            print(f"  {left} <> {doc_ids[j]}: {similarity[i][j]:.3f}")
    for cluster in clusters:
        names = ", ".join(member.doc_id for member in cluster.members)
        print(f"{cluster.cluster_id}: {names}")
        print(f"  top terms: {', '.join(cluster.top_terms)}")
        print(f"  suggested industry: {cluster.suggested_industry or '(mixed or abstained)'}")
        print(f"  suggested document type: {cluster.suggested_doc_type or '(mixed or abstained)'}")
        for member in cluster.members:
            industry = member.industry
            doc_type = member.doc_type
            if industry is None or doc_type is None:
                continue
            print(
                f"  - {member.doc_id}: industry {industry.label} "
                f"({industry.score:.2f}), doc_type {doc_type.label} "
                f"({doc_type.score:.2f})"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotag", description="Classify and cluster a few plain-text documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    classify = sub.add_parser("classify", help="Classify one UTF-8 text file.")
    classify.add_argument("path")
    classify.add_argument("--backend", choices=("stub", "jev"), default="stub")
    classify.add_argument("--endpoint", default="http://127.0.0.1:8017/v1/systemone")
    classify.add_argument("--threshold", type=float, default=0.45)
    classify.add_argument("--json", action="store_true")
    classify.set_defaults(func=_cmd_classify)

    cluster = sub.add_parser("cluster", help="Cluster every .txt file in a directory.")
    cluster.add_argument("directory")
    cluster.add_argument("--backend", choices=("stub", "jev"), default="stub")
    cluster.add_argument("--endpoint", default="http://127.0.0.1:8017/v1/systemone")
    cluster.add_argument("--similarity", type=float, default=0.2, help="Average-linkage merge threshold.")
    cluster.add_argument("--json", action="store_true")
    cluster.set_defaults(func=_cmd_cluster)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
