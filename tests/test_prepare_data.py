from __future__ import annotations

import pytest

from axion.data import normalize_dataset_prefix


def test_dataset_prefix_is_normalized() -> None:
    assert normalize_dataset_prefix("sample/100BT") == "sample/100BT/"
    assert normalize_dataset_prefix("sample/100BT/") == "sample/100BT/"


@pytest.mark.parametrize("prefix", ("/sample/100BT", "sample/../100BT"))
def test_dataset_prefix_rejects_unsafe_paths(prefix: str) -> None:
    with pytest.raises(ValueError, match="invalid dataset prefix"):
        normalize_dataset_prefix(prefix)
