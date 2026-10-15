#!/usr/bin/env python3
"""Generate a request trace file.

Usage:
    python series_generate.py [output_file] [n_requests]

Defaults: output file is time-series.json next to this script, 100 requests.
The output is deterministic (fixed seed).
"""

import argparse
import json
import random
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).parent / "time-series.json"
DEFAULT_COUNT = 100
SEED = 42


def generate(n: int) -> list[dict]:
    rng = random.Random(SEED)
    entries = []
    now = 0.0
    for rid in range(n):
        # arrival gap: mostly moderate, sometimes a burst, sometimes idle
        r = rng.random()
        if r < 0.10:
            gap = rng.uniform(5, 30)        # burst
        elif r < 0.15:
            gap = rng.uniform(2000, 5000)   # idle stretch
        else:
            gap = rng.uniform(50, 500)
        now += gap

        input_tokens = max(128, int(rng.lognormvariate(9.0, 0.7)))
        input_tokens = min(input_tokens, 120_000)
        output_tokens = max(1, int(rng.lognormvariate(6.2, 0.6)))

        cache_hit = 0
        if rng.random() < 0.25:  # some requests share a prefix
            cache_hit = int(input_tokens * rng.uniform(0.2, 0.8))

        entries.append({
            "arrival_ms": round(now, 3),
            "input": input_tokens,
            "output": output_tokens,
            "cache_hit": cache_hit,
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default=str(DEFAULT_OUTPUT),
                        help=f"output file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("count", nargs="?", type=int, default=DEFAULT_COUNT,
                        help=f"number of requests (default: {DEFAULT_COUNT})")
    args = parser.parse_args()

    entries = generate(args.count)
    Path(args.output).write_text(
        json.dumps(entries, indent=1) + "\n", encoding="utf-8")
    span_s = entries[-1]["arrival_ms"] / 1000 if entries else 0
    print(f"wrote {len(entries)} requests to {args.output} "
          f"(span {span_s:.1f}s)")


if __name__ == "__main__":
    main()
