"""Group documents with a small agglomerative clusterer.

The default embedder is lexical term frequency so the pipeline runs offline. A neural
embedder with the same `embed` method can replace it later; that dependency
is documented in the plan and is not added to pyproject.toml.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from autotag.classify import Classification, Classifier, StubClassifier, classify_document

_STOPS = frozenset(
    """
    a an the of and to for within this is or by with on in at from be may
    either that as are was were it its their his her our your not no
    """.split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalized vector per text, in order."""


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPS and len(token) > 1 and not token.isdigit()
    ]


def _l2(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class LexicalTfEmbedder:
    """L2-normalized term frequency over the texts passed to one `embed` call.

    IDF is a poor weighting on a handful of documents: the words that mark a
    group are exactly the words that occur in more than one text, so IDF
    shrinks them. A neural embedder replaces this class later.
    """

    name = "lexical-tf"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        tokenized = [_content_tokens(text) for text in texts]
        vocabulary = sorted({token for tokens in tokenized for token in tokens})
        index = {token: position for position, token in enumerate(vocabulary)}
        vectors: list[list[float]] = []
        for tokens in tokenized:
            vector = [0.0] * len(vocabulary)
            for token, frequency in Counter(tokens).items():
                vector[index[token]] = float(frequency)
            vectors.append(_l2(vector))
        return vectors


@dataclass
class Member:
    doc_id: str
    industry: Classification | None = None
    doc_type: Classification | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"doc_id": self.doc_id}
        if self.industry is not None:
            payload["industry"] = self.industry.to_dict()
        if self.doc_type is not None:
            payload["doc_type"] = self.doc_type.to_dict()
        return payload


@dataclass
class Cluster:
    cluster_id: str
    members: list[Member]
    top_terms: list[str] = field(default_factory=list)
    suggested_industry: str | None = None
    suggested_doc_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "members": [member.to_dict() for member in self.members],
            "top_terms": self.top_terms,
            "suggested_industry": self.suggested_industry,
            "suggested_doc_type": self.suggested_doc_type,
        }


def _average_linkage(sim: list[list[float]], left: set[int], right: set[int]) -> float:
    total = 0.0
    for i in left:
        for j in right:
            total += sim[i][j]
    return total / (len(left) * len(right))


def agglomerative(vectors: Sequence[Sequence[float]], threshold: float) -> list[list[int]]:
    """Average-linkage clustering. Merge while the best similarity is >= threshold."""
    clusters: list[set[int]] = [{index} for index in range(len(vectors))]
    similarity = [[_dot(vectors[i], vectors[j]) for j in range(len(vectors))] for i in range(len(vectors))]
    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_score = -1.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = _average_linkage(similarity, clusters[i], clusters[j])
                pair = (i, j)
                better = score > best_score or (score == best_score and best_pair is not None and pair < best_pair)
                if better:
                    best_score = score
                    best_pair = pair
        if best_pair is None or best_score < threshold:
            break
        left, right = best_pair
        merged = clusters[left] | clusters[right]
        clusters = [group for index, group in enumerate(clusters) if index != left and index != right]
        clusters.append(merged)
    ordered = [sorted(group) for group in clusters]
    ordered.sort(key=lambda group: group[0])
    return ordered


def _top_terms(texts: Sequence[str], limit: int = 5) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_content_tokens(text))
    return [token for token, _ in counts.most_common(limit)]


def _agreed_label(values: list[str]) -> str | None:
    unique = set(values)
    if len(unique) == 1:
        only = unique.pop()
        return None if only == "other" else only
    return None


def cluster_texts(
    items: Sequence[tuple[str, str]],
    embedder: Embedder | None = None,
    classifier: Classifier | None = None,
    threshold: float = 0.2,
    classify_threshold: float = 0.45,
) -> tuple[list[Cluster], list[list[float]]]:
    """Cluster `(doc_id, text)` pairs. Returns clusters and the cosine matrix."""
    model = embedder if embedder is not None else LexicalTfEmbedder()
    tagger = classifier if classifier is not None else StubClassifier()
    ids = [doc_id for doc_id, _ in items]
    texts = [text for _, text in items]
    vectors = model.embed(texts)
    groups = agglomerative(vectors, threshold)
    similarity = [[_dot(vectors[i], vectors[j]) for j in range(len(vectors))] for i in range(len(vectors))]
    clusters: list[Cluster] = []
    for number, group in enumerate(groups):
        members: list[Member] = []
        industries: list[str] = []
        doc_types: list[str] = []
        for index in group:
            tagged = classify_document(texts[index], tagger, classify_threshold)
            members.append(Member(ids[index], tagged["industry"], tagged["doc_type"]))
            industries.append(tagged["industry"].label)
            doc_types.append(tagged["doc_type"].label)
        clusters.append(
            Cluster(
                cluster_id=f"c{number}",
                members=members,
                top_terms=_top_terms([texts[index] for index in group]),
                suggested_industry=_agreed_label(industries),
                suggested_doc_type=_agreed_label(doc_types),
            )
        )
    return clusters, similarity


def cluster_paths(
    paths: Sequence[Path],
    embedder: Embedder | None = None,
    classifier: Classifier | None = None,
    threshold: float = 0.2,
) -> tuple[list[Cluster], list[str], list[list[float]]]:
    items = [(path.name, path.read_text(encoding="utf-8")) for path in paths]
    clusters, similarity = cluster_texts(items, embedder, classifier, threshold)
    return clusters, [doc_id for doc_id, _ in items], similarity
