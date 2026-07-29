from __future__ import annotations

import pytest

from axion.checkpoints import newest_common_checkpoint


def test_newest_common_checkpoint_falls_back_from_partial_save() -> None:
    filenames = [
        ["stage0-step00002000.pt", "stage0-step00004000.pt"],
        ["stage1-step00002000.pt"],
        ["stage2-step00002000.pt"],
    ]

    assert newest_common_checkpoint(filenames) == 2000


def test_newest_common_checkpoint_rejects_disjoint_stages() -> None:
    with pytest.raises(RuntimeError, match="no common distributed checkpoint"):
        newest_common_checkpoint(
            [
                ["stage0-step00004000.pt"],
                ["stage1-step00002000.pt"],
                ["stage2-step00002000.pt"],
            ]
        )
