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
        # prefill 32k tokens @40k tps = 800ms; first chunk (8192 tokens)
        # lands after ~205ms compute + transfer; TTFT must sit in between
        self.assertGreater(summary["avg_ttft_ms"], 150)
        self.assertLess(summary["avg_ttft_ms"], 800)
        # E2E = prefill + 100 decode iterations (~50ms each at batch 1)
        self.assertGreater(summary["avg_e2e_ms"], 5000)

    def test_full_cache_hit_skips_compute(self):
        raw = [{"input": 16000, "output": 10, "cache_hit": 16000,
                "arrival_ms": 0}]
        summary, _ = run_trace(raw)
        self.assertEqual(summary["completed"], 1)
        # no prefill compute at all: TTFT is just transfer latency
        self.assertLess(summary["avg_ttft_ms"], 100)

    def test_chunked_vs_oneshot(self):
        raw = [{"input": 32000, "output": 5, "cache_hit": 0,
                "arrival_ms": 0}]
        chunked, _ = run_trace(raw)
        oneshot, _ = run_trace(raw, chunked_prefill_size=None)
        # one-shot transfers everything at the end -> later TTFT
        self.assertLess(chunked["avg_ttft_ms"], oneshot["avg_ttft_ms"])

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
