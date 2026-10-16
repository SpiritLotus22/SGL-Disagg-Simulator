"""Backpressure: waiting is not rejection; hard rejects only when hopeless."""

import unittest

from simulator import SimConfig, build
from cli import parse_trace


def run_trace(raw, **overrides):
    config = SimConfig(**overrides)
    sim = build(config, parse_trace(raw))
    return sim.run(), sim


class TestBackpressure(unittest.TestCase):
    def test_wait_is_not_reject(self):
        # decode pool fits exactly one 8k prompt; the second must wait
        # for the first to finish, then get in -- nobody is rejected.
        raw = [
            {"input": 8000, "output": 30, "cache_hit": 0, "arrival_ms": 0},
            {"input": 8000, "output": 30, "cache_hit": 0, "arrival_ms": 0},
        ]
        summary, sim = run_trace(
            raw,
            decode_pool_tokens=10_000, decode_kv_watermark=1.0,
            num_reserved_decode_tokens=0, page_size=16,
            clip_max_new_tokens=4096, init_new_token_ratio=0.9,
            min_new_token_ratio=0.9,
            decode_iter_base_ms=10.0, decode_iter_slope_ms=0.0,
        )
        self.assertEqual(summary["decisions"].get("ACCEPT"), 2)
        self.assertNotIn("REJECT_DECODE_KV", summary["decisions"])
        self.assertEqual(summary["completed"], 2)
        # the second request's grant is delayed: positive average wait
        self.assertGreater(summary["avg_wait_ms"], 100)

    def test_never_fits_decode_hard_reject(self):
        raw = [{"input": 50000, "output": 10, "cache_hit": 0,
                "arrival_ms": 0}]
        summary, _ = run_trace(
            raw, decode_pool_tokens=10_000, decode_kv_watermark=1.0,
            num_reserved_decode_tokens=0,
        )
        self.assertEqual(summary["decisions"].get("REJECT_DECODE_KV"), 1)
        self.assertEqual(summary["completed"], 0)
        self.assertEqual(summary["accept_rate"], 0.0)

    def test_accept_rate_counts_arrivals(self):
        # one accepted, one never-fits rejected: accept rate is 1/2
        raw = [
            {"input": 100, "output": 10, "cache_hit": 0, "arrival_ms": 0},
            {"input": 50000, "output": 10, "cache_hit": 0, "arrival_ms": 0},
        ]
        summary, _ = run_trace(
            raw, decode_pool_tokens=10_000, decode_kv_watermark=1.0,
            num_reserved_decode_tokens=0,
        )
        self.assertEqual(summary["accept_rate"], 0.5)

    def test_context_len_hard_reject(self):
        raw = [{"input": 200_000, "output": 10, "cache_hit": 0,
                "arrival_ms": 0}]
        summary, _ = run_trace(raw)  # default context_len 131072
        self.assertEqual(
            summary["decisions"].get("REJECT_PREFILL_PRESSURE"), 1)

    def test_max_running_requests_gates(self):
        raw = [{"input": 100, "output": 1000, "cache_hit": 0,
                "arrival_ms": i} for i in range(4)]
        summary, _ = run_trace(
            raw, max_running_requests=1, decode_iter_base_ms=5.0,
            decode_iter_slope_ms=0.0, prefill_tps=400_000,
            transfer_base_latency_ms=0.1,
        )
        # all accepted eventually (queued, not rejected) and all complete
        self.assertEqual(summary["completed"], 4)
        self.assertNotIn("REJECT_DECODE_KV", summary["decisions"])


if __name__ == "__main__":
    unittest.main()
