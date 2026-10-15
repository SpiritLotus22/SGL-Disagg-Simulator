"""End-to-end lifecycle: one request walks the whole chain."""

import unittest

from simulator import SimConfig, build
from cli import parse_trace


def run_trace(raw, **overrides):
    config = SimConfig(**overrides)
    sim = build(config, parse_trace(raw))
    return sim.run(), sim


class TestLifecycle(unittest.TestCase):
    def test_single_request_full_chain(self):
        raw = [{"input": 32000, "output": 100, "cache_hit": 0,
                "arrival_ms": 0}]
        summary, sim = run_trace(raw)
        self.assertEqual(summary["decisions"].get("ACCEPT"), 1)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["retracts"], 0)
        # TTFT = arrival -> last KV chunk at decode: full prefill (32k
        # @40k tps = 800ms) + final chunk transfer (~23ms)
        self.assertGreater(summary["avg_ttft_ms"], 800)
        self.assertLess(summary["avg_ttft_ms"], 1200)
        # the first KV chunk arrives DURING prefill, long before TTFT
        self.assertLess(summary["avg_first_kv_ms"], 400)
        self.assertGreater(summary["avg_ttft_ms"],
                           summary["avg_first_kv_ms"])
        # E2E = prefill + 100 decode iterations (~50ms each at batch 1)
        self.assertGreater(summary["avg_e2e_ms"], 5000)

    def test_full_cache_hit_skips_compute(self):
        raw = [{"input": 16000, "output": 10, "cache_hit": 16000,
                "arrival_ms": 0}]
        summary, _ = run_trace(raw)
        self.assertEqual(summary["completed"], 1)
        # no prefill compute: cached KV streams at once, TTFT is just
        # the transfer of a single chunk
        self.assertLess(summary["avg_ttft_ms"], 100)
        self.assertEqual(summary["avg_ttft_ms"],
                         summary["avg_first_kv_ms"])  # one chunk only

    def test_chunked_vs_oneshot(self):
        raw = [{"input": 32000, "output": 5, "cache_hit": 0,
                "arrival_ms": 0}]
        chunked, _ = run_trace(raw)
        oneshot, _ = run_trace(raw, chunked_prefill_size=None)
        # both finish computing at the same time, but one-shot then has
        # to ship the whole prompt in a single huge transfer -> later TTFT
        self.assertLess(chunked["avg_ttft_ms"], oneshot["avg_ttft_ms"])
        # the difference is far more visible in the first-chunk latency
        self.assertLess(chunked["avg_first_kv_ms"],
                        oneshot["avg_first_kv_ms"] / 2)

    def test_example_trace(self):
        from pathlib import Path
        from cli import load_trace
        trace = load_trace(Path(__file__).parent.parent.joinpath(
            "examples", "example.json"))
        # the 140k-token prompt exceeds the default context_len
        sim = build(SimConfig(), trace)
        summary = sim.run()
        self.assertEqual(summary["requests"], 12)
        self.assertEqual(
            summary["decisions"].get("REJECT_PREFILL_PRESSURE"), 1)
        self.assertEqual(summary["completed"], 11)


if __name__ == "__main__":
    unittest.main()
