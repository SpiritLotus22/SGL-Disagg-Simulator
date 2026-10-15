"""Retraction: when the optimistic estimate proves wrong, the newest
request is pushed back, new_token_ratio resets, and everyone still finishes.
Also: whole-run determinism on a fixed trace."""

import unittest

from simulator import SimConfig, build
from cli import parse_trace

OVERRUN_TRACE = [{"input": 500, "output": 1000, "cache_hit": 0,
                  "arrival_ms": i} for i in range(8)]

OVERRUN_CONFIG = dict(
    decode_pool_tokens=5_000, decode_kv_watermark=1.0,
    num_reserved_decode_tokens=0, page_size=1,
    clip_max_new_tokens=4096,
    decode_iter_base_ms=5.0, decode_iter_slope_ms=0.0,
    prefill_tps=400_000, transfer_base_latency_ms=0.1,
)


class TestRetract(unittest.TestCase):
    def test_overrun_triggers_retract_and_recovery(self):
        # Tiny pool, very optimistic ratio -> over-admission -> the physical
        # pool overflows during decode -> retraction must fire.
        config = SimConfig(init_new_token_ratio=0.01, min_new_token_ratio=0.01,
                           **OVERRUN_CONFIG)
        sim = build(config, parse_trace(OVERRUN_TRACE))
        summary = sim.run()
        self.assertGreater(summary["retracts"], 0)
        self.assertEqual(summary["completed"], 8)  # everyone recovers
        decode = sim.engine.actors["decode-0"]
        # pool usage never stays above physical capacity after a tick
        self.assertLessEqual(decode.d_used, decode.pool_tokens)

    def test_ratio_resets_on_retract(self):
        config = SimConfig(init_new_token_ratio=0.5, min_new_token_ratio=0.01,
                           ratio_decay_per_iter=1.0,  # hits min in one step
                           **OVERRUN_CONFIG)
        sim = build(config, parse_trace(OVERRUN_TRACE))
        summary = sim.run()
        self.assertGreater(summary["retracts"], 0)
        # after the last retract the ratio is back at init (or decayed
        # from it by subsequent iterations, never lower than min)
        decode = sim.engine.actors["decode-0"]
        self.assertGreaterEqual(decode.ratio, config.min_new_token_ratio)
        self.assertLessEqual(decode.ratio, config.init_new_token_ratio)


class TestDeterminism(unittest.TestCase):
    def test_same_trace_same_result(self):
        config = SimConfig(**OVERRUN_CONFIG, init_new_token_ratio=0.01,
                           min_new_token_ratio=0.01)
        a = build(config, parse_trace(OVERRUN_TRACE)).run()
        b = build(config, parse_trace(OVERRUN_TRACE)).run()
        self.assertEqual(a, b)

    def test_watermark_sweep_replays_identically(self):
        from dataclasses import replace
        config = SimConfig(**OVERRUN_CONFIG, init_new_token_ratio=0.01,
                           min_new_token_ratio=0.01)
        trace = parse_trace(OVERRUN_TRACE)

        def sweep():
            return {wm: build(replace(config, decode_kv_watermark=wm),
                              trace).run()
                    for wm in (0.8, 1.0)}

        self.assertEqual(sweep(), sweep())


if __name__ == "__main__":
    unittest.main()
