#!/usr/bin/env python3
"""Generate a heavier mixed trace for admission-control demos.

Usage:
    python heavy_mixed_generate.py [output_file] [n_requests]

Defaults: output file is heavy-mixed.json next to this script, 160 requests.
The output is deterministic (fixed seed).

Composition (three request classes):
  - medium (60%): lognormal input around ~7k tokens, output around ~400;
  - long-cold (25%): 40k-90k token prompts, no cache, long outputs;
  - cached (15%): 20k-50k prompts with 50%-80% prefix hits.
Arrival gaps: mostly 150-450ms, with bursts and pauses mixed in, so the
decode pool hovers near saturation instead of being instantly flooded.
"""

import argparse
import json
import random
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).parent / "heavy-mixed.json"
DEFAULT_COUNT = 160
SEED = 7


def generate(n: int) -> list[dict]:
    rng = random.Random(SEED)
    entries = []
    now = 0.0
    for rid in range(n):
        r = rng.random()
        if r < 0.15:
            gap = rng.uniform(10, 60)       # burst
        elif r < 0.30:
            gap = rng.uniform(500, 1200)    # lull
        else:
            gap = rng.uniform(150, 450)
        now += gap

        kind = rng.random()
        if kind < 0.60:                     # medium
            input_tokens = max(128, int(rng.lognormvariate(8.8, 0.6)))
            input_tokens = min(input_tokens, 32_000)
            output_tokens = max(1, int(rng.lognormvariate(6.0, 0.5)))
            cache_hit = 0
        elif kind < 0.85:                   # long-cold
            input_tokens = rng.randint(40_000, 90_000)
            output_tokens = max(1, int(rng.lognormvariate(7.3, 0.5)))
            cache_hit = 0
        else:                               # cached
            input_tokens = rng.randint(20_000, 50_000)
            cache_hit = int(input_tokens * rng.uniform(0.5, 0.8))
            output_tokens = max(1, int(rng.lognormvariate(6.0, 0.5)))

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
