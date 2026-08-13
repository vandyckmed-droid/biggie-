#!/usr/bin/env python3
"""Rebuild the snapshot from the command line.

    python scripts/refresh.py                 # incremental (only new bars)
    python scripts/refresh.py --full          # re-download the full history
    python scripts/refresh.py --size 250      # smaller universe, useful for a quick check
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config, snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the Biggie snapshot")
    parser.add_argument("--size", type=int, default=config.UNIVERSE_SIZE,
                        help="number of stocks in the universe")
    parser.add_argument("--k", type=int, default=config.DEFAULT_CLUSTER_K,
                        help="cluster count for medoid clustering")
    parser.add_argument("--full", action="store_true",
                        help="ignore the price cache and refetch everything")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    started = time.time()
    last = [""]

    def progress(stage: str, pct: float) -> None:
        if args.quiet:
            return
        bar = "#" * int(pct * 28)
        line = f"\r  [{bar:<28}] {pct * 100:5.1f}%  {stage[:48]:<48}"
        if line != last[0]:
            sys.stdout.write(line)
            sys.stdout.flush()
            last[0] = line

    try:
        data = asyncio.run(
            snapshot.full_refresh(
                size=args.size, cluster_k=args.k, progress=progress,
                force_full=args.full,
            )
        )
    except Exception as exc:
        print(f"\nRefresh failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print()

    report = data["universe_report"]
    regime = data["macro"]["regime"]
    print(f"Snapshot written to {config.SNAPSHOT_PATH}")
    print(f"  as of          {data['as_of']}  ({data['trading_days']} trading days)")
    print(f"  universe       {data['universe_size']} of {report['screened']} screened "
          f"({report['deduped']} duplicate listings collapsed)")
    if report.get("excluded"):
        detail = ", ".join(f"{k}={v}" for k, v in sorted(report["excluded"].items()))
        print(f"  excluded       {detail}")
    print(f"  shrinkage      {data['risk_model']['shrinkage']:.3f} "
          f"over {data['risk_model']['observations']} observations")
    print(f"  regime         {regime['state']} ({regime['score']:+.2f})")
    print(f"  HRP vol        {data['hrp_universe']['portfolio_vol']:.2%} "
          f"(effective N = {data['hrp_universe']['effective_n']:.0f})")
    print(f"  total time     {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
