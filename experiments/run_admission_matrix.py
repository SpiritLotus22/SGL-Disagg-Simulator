#!/usr/bin/env python3
"""准入预算 × decode 水位 矩阵实验。

对两份 trace（heavy = examples/heavy-mixed.json 重过载，
moderate = examples/time-series.json 轻度过载），以 scenarios/2p2d.json
为基础配置，扫描 6 档准入预算（off = 纯 backpressure，以及 1s/2s/5s/10s/20s
双闸门同值）× 5 档 decode 水位，共 30 个组合。

结果写入 experiments/results/admission_matrix_{heavy,moderate}.{json,md}。

用法：
    python experiments/run_admission_matrix.py            # 两份 trace 全跑
    python experiments/run_admission_matrix.py heavy      # 只跑一份
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import load_trace
from simulator import SimConfig, build

WATERMARKS = (0.60, 0.70, 0.80, 0.90, 0.95)
# None = 准入关闭（纯 backpressure），相当于预算正无穷，排在最后
BUDGETS_MS = (1000, 2000, 5000, 10000, 20000, None)

TRACES = {
    "heavy": ROOT / "examples" / "heavy-mixed.json",
    "moderate": ROOT / "examples" / "time-series.json",
}

RESULTS_DIR = Path(__file__).parent / "results"


def run_one(config: SimConfig, trace: list, budget_ms, watermark: float) -> dict:
    cfg = replace(config, decode_kv_watermark=watermark)
    if budget_ms is not None:
        cfg = replace(cfg, max_bootstrap_wait_ms=budget_ms,
                      prefill_max_queue_wait_ms=budget_ms)
    return build(cfg, trace).run()


def to_row(budget_ms, watermark: float, s: dict) -> dict:
    dec = s["decisions"]
    return {
        "budget_ms": budget_ms,
        "watermark": watermark,
        "accept": dec.get("ACCEPT", 0),
        "accept_rate": s["accept_rate"],
        "reject_prefill": dec.get("REJECT_PREFILL_PRESSURE", 0)
        + dec.get("REJECT_PREFILL_WAIT", 0),
        "reject_decode": dec.get("REJECT_DECODE_KV", 0)
        + dec.get("REJECT_DECODE_WAIT", 0),
        "retracts": s["retracts"],
        "peak_prefill_util": s["peak_prefill_util"],
        "peak_decode_util": s["peak_decode_util"],
        "avg_wait_ms": s["avg_wait_ms"],
        "avg_ttft_ms": s["avg_ttft_ms"],
        "p95_ttft_ms": s["p95_ttft_ms"],
        "avg_e2e_ms": s["avg_e2e_ms"],
        "p95_e2e_ms": s["p95_e2e_ms"],
    }


def format_table(rows: list[dict]) -> str:
    header = (
        f"{'budget':>7} | {'wm':>4} | {'ACCEPT':>6} | {'acc%':>5} | "
        f"{'RJ_P':>4} | {'RJ_D':>4} | {'retract':>7} | {'peak P':>7} | "
        f"{'peak D':>7} | "
        f"{'wait ms':>9} | {'TTFT avg':>9} | {'TTFT p95':>9} | "
        f"{'E2E p95':>9}")
    lines = [header, "-" * len(header)]
    for r in rows:
        budget = "off" if r["budget_ms"] is None else f"{r['budget_ms']}"
        lines.append(
            f"{budget:>7} | {r['watermark']:>4.0%} | {r['accept']:>6} | "
            f"{r['accept_rate']:>5.0%} | "
            f"{r['reject_prefill']:>4} | {r['reject_decode']:>4} | "
            f"{r['retracts']:>7} | {r['peak_prefill_util']:>6.1%} | "
            f"{r['peak_decode_util']:>6.1%} | {r['avg_wait_ms']:>9.1f} | "
            f"{r['avg_ttft_ms']:>9.1f} | {r['p95_ttft_ms']:>9.1f} | "
            f"{r['p95_e2e_ms']:>9.1f}")
    return "\n".join(lines)


def run_trace(name: str, trace_path: Path) -> list[dict]:
    base = SimConfig.from_dict(
        json.loads((ROOT / "scenarios" / "2p2d.json").read_text(
            encoding="utf-8")))
    trace = load_trace(trace_path)
    rows = []
    for budget in BUDGETS_MS:
        for wm in WATERMARKS:
            summary = run_one(base, trace, budget, wm)
            rows.append(to_row(budget, wm, summary))
            label = "off" if budget is None else f"{budget}ms"
            print(f"[{name}] budget={label:>6} wm={wm:.2f} "
                  f"accept={rows[-1]['accept']:>3} "
                  f"rj_d={rows[-1]['reject_decode']:>3} "
                  f"p95_ttft={rows[-1]['p95_ttft_ms']:>10.1f}ms",
                  flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"admission_matrix_{name}.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"admission_matrix_{name}.md").write_text(
        format_table(rows) + "\n", encoding="utf-8")
    print(f"[{name}] -> {RESULTS_DIR}/admission_matrix_{name}.json/.md")
    return rows


def main(argv=None) -> int:
    names = argv if argv else list(TRACES)
    for name in names:
        run_trace(name, TRACES[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
