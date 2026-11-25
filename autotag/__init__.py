"""AutoTag: open industry phrase plus a fixed document type.

Not part of the Axion training milestones. The default path is the standard
library. OpenJev weights load only when that backend is requested.
"""

from autotag.classify import Classification, classify_document
from autotag.cluster import Cluster, cluster_paths

__all__ = [
    "Classification",
    "Cluster",
    "classify_document",
    "cluster_paths",
]
