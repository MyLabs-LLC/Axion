"""AutoTag: document industry and document-type exploration.

This package is not part of the Axion training milestones. It uses the
standard library only. A real CPU model is swapped in behind Classifier.
"""

from autotag.classify import Classification, classify_document
from autotag.cluster import Cluster, cluster_paths

__all__ = [
    "Classification",
    "Cluster",
    "classify_document",
    "cluster_paths",
]
