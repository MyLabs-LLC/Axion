from __future__ import annotations


def checkpoint_step(filename: str, rank: int) -> int:
    prefix = f"stage{rank}-step"
    if not filename.startswith(prefix) or not filename.endswith(".pt"):
        raise ValueError(f"invalid stage checkpoint name: {filename}")
    return int(filename[len(prefix) : -3])


def newest_common_checkpoint(filenames: list[list[str]]) -> int | None:
    checkpoint_sets = [
        {checkpoint_step(name, rank) for name in names}
        for rank, names in enumerate(filenames)
    ]
    if all(not steps for steps in checkpoint_sets):
        return None
    common = set.intersection(*checkpoint_sets)
    if not common:
        latest = [max(steps, default=None) for steps in checkpoint_sets]
        raise RuntimeError(f"no common distributed checkpoint; latest stage steps: {latest}")
    return max(common)
