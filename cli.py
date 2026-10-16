#!/usr/bin/env python3
"""Command line entry point.

Examples:
    python cli.py --input examples/example.json
    python cli.py --input examples/example.json \
        --config scenarios/example_trace.json --watermarks 0.7 0.9
    python cli.py --input examples/example.json --trace
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from protocol import (
    DecodeHeartbeat, DecodeTick, Heartbeat, PrefillTick, RequestSpec,
)
from simulator import SimConfig, build

DEFAULT_WATERMARKS = (0.60, 0.70, 0.80, 0.90, 0.95)

# These message types dominate the event stream and rarely help when
# reading a trace; each is hidden unless the scenario opts back in.
_QUIET_FLAGS = (
    ("trace_print_prefill_tick", PrefillTick),
    ("trace_print_decode_tick", DecodeTick),
    ("trace_print_heartbeat", Heartbeat),
    ("trace_print_decode_heartbeat", DecodeHeartbeat),
)


def trace_hidden_types(config: SimConfig) -> frozenset:
    """Message types excluded from --trace output under this config."""
    return frozenset(t for flag, t in _QUIET_FLAGS
                     if not getattr(config, flag))


def parse_request(data: dict, rid: int) -> tuple[float, "RequestSpec"]:
    """Validate one trace entry and return (arrival_ms, RequestSpec)."""
    if not isinstance(data, dict):
        raise ValueError(f"request {rid} is not an object: {data!r}")
    for key in ("arrival_ms", "input", "output"):
        if key not in data:
            raise ValueError(f"request {rid} is missing {key!r}: {data}")
    try:
        time_ms = float(data["arrival_ms"])
        input_tokens = int(data["input"])
        output_tokens = int(data["output"])
        cache_hit = int(data.get("cache_hit", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"request {rid} has a non-numeric field: {data} "
                         f"({exc})") from exc
    if time_ms < 0:
        raise ValueError(f"negative arrival_ms in request {rid}: {data}")
    if input_tokens < 0 or output_tokens < 0 or cache_hit < 0:
        raise ValueError(f"negative token count in request {rid}: {data}")
    if cache_hit > input_tokens:
        raise ValueError(
            f"request {rid}: cache_hit ({cache_hit}) exceeds input "
            f"({input_tokens})"
        )
    return time_ms, RequestSpec(
        rid=rid, input_tokens=input_tokens,
        output_tokens=output_tokens, cache_hit=cache_hit,
    )


def parse_trace(raw: list[dict]) -> list:
    """Validate a whole trace; returns [(arrival_ms, RequestSpec), ...]."""
    return [parse_request(item, rid) for rid, item in enumerate(raw)]


def load_trace(path) -> list:
    """Read a trace JSON file and validate it."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_trace(raw)


def format_sweep(results: dict[float, dict]) -> str:
    header = (f"{'watermark':>9} | {'ACCEPT':>6} | {'RJ_P':>4} | {'RJ_D':>4} | "
              f"{'retract':>7} | {'peak P':>7} | {'peak D':>7} | "
              f"{'wait ms':>9} | {'TTFT p95':>9} | {'E2E p95':>9}")
    lines = [header, "-" * len(header)]
    for wm, s in results.items():
        dec = s["decisions"]
        lines.append(
            f"{wm:>9.0%} | {dec.get('ACCEPT', 0):>6} | "
            f"{dec.get('REJECT_PREFILL_PRESSURE', 0):>4} | "
            f"{dec.get('REJECT_DECODE_KV', 0):>4} | "
            f"{s['retracts']:>7} | {s['peak_prefill_util']:>6.1%} | "
            f"{s['peak_decode_util']:>6.1%} | {s['avg_wait_ms']:>9.1f} | "
            f"{s['p95_ttft_ms']:>9.1f} | {s['p95_e2e_ms']:>9.1f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a request trace through a PD-disaggregated "
                    "serving system")
    parser.add_argument("--input", required=True,
                        help="request trace file (JSON)")
    parser.add_argument("--config",
                        help="scenario config file (JSON), optional")
    parser.add_argument("--watermarks", nargs="+", type=float, default=None,
                        help="decode KV watermarks to sweep "
                             "(default: 0.6 0.7 0.8 0.9 0.95)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for the scheduling policy")
    parser.add_argument("--trace", action="store_true",
                        help="print every event (single watermark only)")
    args = parser.parse_args(argv)

    scenario = {}
    if args.config:
        scenario = json.loads(Path(args.config).read_text(encoding="utf-8"))
    # seed is a run parameter, not part of the scenario
    config = replace(SimConfig.from_dict(scenario), seed=args.seed)

    trace = load_trace(args.input)
    watermarks = tuple(args.watermarks or DEFAULT_WATERMARKS)

    if len(watermarks) == 1:
        # A full event trace is far too long to print for several runs, so
        # --trace is only supported when sweeping a single watermark. The
        # report goes to a file under traces/, not stdout.
        sim = build(config, trace)
        if args.trace:
            traces_dir = Path(__file__).parent / "traces"
            traces_dir.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = traces_dir / f"trace_{stamp}.txt"
            hidden = trace_hidden_types(config)
            with open(report_path, "w", encoding="utf-8") as fh:
                def record(now, ev):
                    if type(ev.message) in hidden:
                        return
                    fh.write(
                        f"{now:>12.2f} ms  -> {ev.target:<10} "
                        f"{type(ev.message).__name__} {ev.message}\n")
                sim.engine.add_observer(record)
                summary = sim.run()
            print(f"event trace written to {report_path}")
        else:
            summary = sim.run()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        if args.trace:
            raise SystemExit(
                "--trace is only supported with a single watermark "
                "(pass exactly one value to --watermarks)"
            )
        results = {}
        for wm in watermarks:
            results[wm] = build(
                replace(config, decode_kv_watermark=wm), trace).run()
        print(format_sweep(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
