"""Industry phrases drawn from the document, not from a taxonomy.

An entailment model cannot invent a string it was not shown. Candidates are
spans that already occur in the text. The model accepts one of them or abstains.
Nothing in this module is a NAICS code or a stored industry list.
"""

from __future__ import annotations

import re
from collections import Counter

from autotag.labels import DOC_TYPE_TOKENS

INDUSTRY_HYPOTHESIS = "The industry this document belongs to is {phrase}."

_STOPS = frozenset(
    """
    a an the of and to for within this is or by with on in at from be may
    either that as are was were it its their his her our your not no
    usd paid due date dates description account term terms between
    """.split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_TITLE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPS and token not in DOC_TYPE_TOKENS and len(token) > 2 and not token.isdigit()
    ]


def _usable(phrase: str) -> bool:
    parts = _tokens(phrase)
    if not parts:
        return False
    raw = phrase.split()
    if any(part.lower() in DOC_TYPE_TOKENS for part in raw):
        return False
    return True


def industry_candidates(text: str, limit: int = 12) -> list[str]:
    """Title spans first, then frequent bigrams, then frequent unigrams."""
    found: list[str] = []
    seen: set[str] = set()

    def add(phrase: str) -> None:
        cleaned = " ".join(phrase.split())
        key = cleaned.casefold()
        if not cleaned or key in seen or not _usable(cleaned):
            return
        seen.add(key)
        found.append(cleaned)

    for match in _TITLE.findall(text):
        add(match)

    tokens = _tokens(text)
    bigrams = Counter(" ".join(pair) for pair in zip(tokens, tokens[1:]))
    for phrase, _ in bigrams.most_common(8):
        add(phrase)
    unigrams = Counter(tokens)
    for phrase, _ in unigrams.most_common(8):
        add(phrase)
    return found[:limit]


def industry_hypothesis(phrase: str) -> str:
    return INDUSTRY_HYPOTHESIS.format(phrase=phrase)
