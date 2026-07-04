#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from axion.metrics import read_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    rows = read_metrics(Path(args.run))
    if not rows:
        raise SystemExit("no metrics found")
    last = rows[-1]
    avg_tps = sum(float(r.get("tokens_per_s", 0.0)) for r in rows) / len(rows)
    print(f"run: {args.run}")
    print(f"steps: {len(rows)}")
    print(f"last_loss: {last.get('loss'):.4f}")
    print(f"avg_tokens_per_s: {avg_tps:.1f}")
    print(f"bytes_tx_step0: {rows[0].get('bytes_tx', 0)}")
    print(f"bytes_rx_step0: {rows[0].get('bytes_rx', 0)}")
    print(f"bubble_frac: {last.get('bubble_frac', 0.0)}")


if __name__ == "__main__":
    main()
