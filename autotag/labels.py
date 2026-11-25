"""Closed label spaces for industry and document type.

Industry is one name from INDUSTRIES. Document type is one name from DOC_TYPES.
Each label has one yes/no question. OpenJev scores the declarative hypothesis,
which is that question with the answer yes. Abstention is a threshold in the
caller, not an extra question.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    question: str
    hypothesis: str


def _industry(id_: str, name: str, gloss: str) -> Label:
    question = f"Is this document mainly about the {name} industry ({gloss})?"
    hypothesis = f"This document is mainly about the {name} industry ({gloss})."
    return Label(id_, name, question, hypothesis)


def _doc_type(id_: str, name: str, question: str, hypothesis: str) -> Label:
    return Label(id_, name, question, hypothesis)


# Sixteen common industries. Names are the labels. This is not a NAICS encoding.
INDUSTRIES: tuple[Label, ...] = (
    _industry("healthcare", "healthcare", "hospitals, clinics, patients, or medical care"),
    _industry("software", "software", "software products, SaaS, or hosted software platforms"),
    _industry("finance", "finance", "banking, lending, or investment"),
    _industry("insurance", "insurance", "insurance policies, premiums, or claims"),
    _industry("legal", "legal", "law firms, attorneys, or legal services"),
    _industry("education", "education", "schools, universities, or teaching"),
    _industry("energy", "energy", "electricity, oil, gas, or utilities"),
    _industry("manufacturing", "manufacturing", "factories or making physical goods"),
    _industry("retail", "retail", "selling goods to consumers in stores"),
    _industry("transportation", "transportation", "freight, shipping, or passenger transport"),
    _industry("real_estate", "real estate", "property sales, leasing, or real estate"),
    _industry("construction", "construction", "building construction or contractors"),
    _industry("government", "government", "government or public administration"),
    _industry("hospitality", "hospitality", "hotels, restaurants, or food service"),
    _industry("telecommunications", "telecommunications", "telephone, internet access, or broadcasting"),
    _industry("agriculture", "agriculture", "farming, crops, or livestock"),
)

DOC_TYPES: tuple[Label, ...] = (
    _doc_type(
        "invoice",
        "Invoice",
        "Is this document an invoice asking for payment?",
        "This document is an invoice asking for payment.",
    ),
    _doc_type(
        "receipt",
        "Receipt",
        "Is this document a receipt confirming a payment already made?",
        "This document is a receipt confirming a payment already made.",
    ),
    _doc_type(
        "contract",
        "Contract",
        "Is this document a contract or agreement between parties?",
        "This document is a contract or agreement between parties.",
    ),
    _doc_type(
        "policy",
        "Policy",
        "Is this document a policy that states rules someone must follow?",
        "This document is a policy that states rules someone must follow.",
    ),
    _doc_type(
        "report",
        "Report",
        "Is this document a report that presents findings or results?",
        "This document is a report that presents findings or results.",
    ),
    _doc_type(
        "letter",
        "Letter",
        "Is this document a letter addressed to a person or organisation?",
        "This document is a letter addressed to a person or organisation.",
    ),
    _doc_type(
        "email",
        "Email",
        "Is this document an email message with a sender and a subject?",
        "This document is an email message with a sender and a subject.",
    ),
    _doc_type(
        "memo",
        "Memo",
        "Is this document an internal memorandum to colleagues?",
        "This document is an internal memorandum to colleagues.",
    ),
    _doc_type(
        "specification",
        "Specification",
        "Is this document a specification of requirements for a product or system?",
        "This document is a specification of requirements for a product or system.",
    ),
    _doc_type(
        "financial_statement",
        "Financial statement",
        "Is this document a financial statement such as a balance sheet, income statement, or cash flow statement?",
        "This document is a financial statement such as a balance sheet, income statement, or cash flow statement.",
    ),
)

ABSTAIN = "other"

INDUSTRY_IDS: frozenset[str] = frozenset(label.id for label in INDUSTRIES)


def industry_hypotheses() -> dict[str, str]:
    return {label.id: label.hypothesis for label in INDUSTRIES}


def industry_names() -> dict[str, str]:
    return {label.id: label.name for label in INDUSTRIES}


def industry_questions() -> dict[str, str]:
    return {label.id: label.question for label in INDUSTRIES}


def doc_type_hypotheses() -> dict[str, str]:
    return {label.id: label.hypothesis for label in DOC_TYPES}


def doc_type_names() -> dict[str, str]:
    return {label.id: label.name for label in DOC_TYPES}
