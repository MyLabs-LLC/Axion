"""Document-type label space. Industry is not a fixed menu.

Document type is one of ten statements. The model score is how strongly the
text entails that statement. Industry is proposed from the document itself
(see industry.py): this file does not list industries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocType:
    id: str
    name: str
    hypothesis: str


DOC_TYPES: tuple[DocType, ...] = (
    DocType("invoice", "Invoice", "This document is an invoice asking for payment."),
    DocType("receipt", "Receipt", "This document is a receipt confirming a payment already made."),
    DocType("contract", "Contract", "This document is a contract or agreement between parties."),
    DocType("policy", "Policy", "This document is a policy that states rules someone must follow."),
    DocType("report", "Report", "This document is a report that presents findings or results."),
    DocType("letter", "Letter", "This document is a letter addressed to a person or organisation."),
    DocType("email", "Email", "This document is an email message with a sender and a subject."),
    DocType("memo", "Memo", "This document is an internal memorandum to colleagues."),
    DocType(
        "specification",
        "Specification",
        "This document is a specification of requirements for a product or system.",
    ),
    DocType(
        "financial_statement",
        "Financial statement",
        "This document is a financial statement such as a balance sheet, income statement, or cash flow statement.",
    ),
)

ABSTAIN = "other"

# Tokens that name a document type. Industry candidates skip these so the
# industry phrase is not just an echo of the type label.
DOC_TYPE_TOKENS: frozenset[str] = frozenset(
    {
        "invoice",
        "receipt",
        "contract",
        "agreement",
        "policy",
        "report",
        "letter",
        "email",
        "memo",
        "memorandum",
        "specification",
        "financial",
        "statement",
        "balance",
        "income",
        "cash",
        "flow",
    }
)

