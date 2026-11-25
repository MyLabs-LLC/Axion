"""Closed label spaces for industry (NAICS 2022 sector) and document type.

Questions are written the way a Jev `noul` item expects: one yes/no question
per label, parallel wording, no negative "other" question. Abstention is a
threshold in the caller, not a label the model is asked to entail.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    question: str


# NAICS 2022 two-digit sectors. Ranges are the official sector codes.
_INDUSTRY_GLOSSES: tuple[tuple[str, str, str], ...] = (
    ("11", "Agriculture, Forestry, Fishing and Hunting", "farms, crops, livestock, fishing, or forestry"),
    ("21", "Mining, Quarrying, and Oil and Gas Extraction", "mining, quarrying, oil, or gas extraction"),
    ("22", "Utilities", "electric, gas, or water utilities"),
    ("23", "Construction", "construction of buildings or infrastructure"),
    ("31-33", "Manufacturing", "manufacturing of physical goods in a plant or factory"),
    ("42", "Wholesale Trade", "wholesale trade of goods to other businesses"),
    ("44-45", "Retail Trade", "retail sale of goods to consumers"),
    ("48-49", "Transportation and Warehousing", "transportation, freight, or warehousing"),
    ("51", "Information", "software, publishing, broadcasting, telecommunications, or data services"),
    ("52", "Finance and Insurance", "banking, lending, investment, or insurance"),
    ("53", "Real Estate and Rental and Leasing", "real estate, rental, or leasing"),
    ("54", "Professional, Scientific, and Technical Services", "legal, accounting, consulting, or scientific services"),
    ("55", "Management of Companies and Enterprises", "a holding company managing other enterprises"),
    (
        "56",
        "Administrative and Support and Waste Management and Remediation Services",
        "administrative support, staffing, or waste management",
    ),
    ("61", "Educational Services", "schools, universities, or other educational services"),
    ("62", "Health Care and Social Assistance", "hospitals, clinics, patients, or medical care"),
    ("71", "Arts, Entertainment, and Recreation", "arts, entertainment, or recreation"),
    ("72", "Accommodation and Food Services", "hotels, restaurants, or food service"),
    ("81", "Other Services (except Public Administration)", "repair, personal care, or civic and membership organizations"),
    ("92", "Public Administration", "government or public administration"),
)

_DOC_TYPE_QUESTIONS: tuple[tuple[str, str, str], ...] = (
    ("invoice", "Invoice", "Is this document an invoice asking for payment?"),
    ("receipt", "Receipt", "Is this document a receipt confirming a payment already made?"),
    ("contract", "Contract", "Is this document a contract or agreement between parties?"),
    ("policy", "Policy", "Is this document a policy that states rules someone must follow?"),
    ("report", "Report", "Is this document a report that presents findings or results?"),
    ("letter", "Letter", "Is this document a letter addressed to a person or organisation?"),
    ("email", "Email", "Is this document an email message with a sender and a subject?"),
    ("memo", "Memo", "Is this document an internal memorandum to colleagues?"),
    ("specification", "Specification", "Is this document a specification of requirements for a product or system?"),
    (
        "financial_statement",
        "Financial statement",
        "Is this document a financial statement such as a balance sheet, income statement, or cash flow statement?",
    ),
)


def _industry_question(name: str, gloss: str) -> str:
    return f"Is this document mainly about the {name} industry ({gloss})?"


INDUSTRIES: tuple[Label, ...] = tuple(
    Label(id_, name, _industry_question(name, gloss)) for id_, name, gloss in _INDUSTRY_GLOSSES
)
DOC_TYPES: tuple[Label, ...] = tuple(Label(id_, name, question) for id_, name, question in _DOC_TYPE_QUESTIONS)

AXES: dict[str, tuple[Label, ...]] = {
    "industry": INDUSTRIES,
    "doc_type": DOC_TYPES,
}

ABSTAIN = "other"


def labels_for(axis: str) -> tuple[Label, ...]:
    try:
        return AXES[axis]
    except KeyError as exc:
        known = ", ".join(sorted(AXES))
        raise KeyError(f"Unknown axis {axis!r}. Expected one of: {known}.") from exc


def questions_for(axis: str) -> dict[str, str]:
    """Label id -> yes/no question. Insertion order follows the taxonomy."""
    return {label.id: label.question for label in labels_for(axis)}


def label_name(axis: str, label_id: str) -> str:
    if label_id == ABSTAIN:
        return "Other"
    for label in labels_for(axis):
        if label.id == label_id:
            return label.name
    return label_id
