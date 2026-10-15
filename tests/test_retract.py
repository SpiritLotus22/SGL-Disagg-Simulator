"""Retraction: when the optimistic estimate proves wrong, requests are
pushed back to prefill (KV-incomplete ones first), retries must clear a
larger headroom, and everyone still finishes. Also: whole-run determinism
on a fixed trace."""

import unittest

from engine import Context, Engine
from perf import DecodePerf
from actors.decode import DecodeInstance
from protocol import BootstrapRequest, KvChunk, RequestSpec
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

    def test_inflight_request_is_retracted_first(self):
        # A decodes while B is still prefilling. When A overruns the pool,
        # B's prealloc is freed (not A's decode progress), B's partial
        # prefill compute is accounted as rework, and both finish.
        trace = parse_trace([
            {"input": 10, "output": 2500, "cache_hit": 0, "arrival_ms": 0},
            {"input": 800, "output": 50, "cache_hit": 0, "arrival_ms": 5},
        ])
        config = SimConfig(
            decode_pool_tokens=3_000, decode_kv_watermark=1.0,
            num_reserved_decode_tokens=0, page_size=1,
            clip_max_new_tokens=4096,
            decode_iter_base_ms=1.0, decode_iter_slope_ms=0.0,
            init_new_token_ratio=0.01, min_new_token_ratio=0.01,
            prefill_tps=100, transfer_base_latency_ms=0.1,
        )
        sim = build(config, trace)
        summary = sim.run()
        self.assertEqual(sim.collector.retracts.get(0, 0), 0)  # A untouched
        self.assertGreater(sim.collector.retracts.get(1, 0), 0)  # B pushed back
        self.assertGreater(summary["retract_rework_tokens"], 0)
        self.assertEqual(summary["completed"], 2)


class TestRetryHysteresis(unittest.TestCase):
    """A retracted retry needs retract_retry_margin x its prealloc in
    headroom (capped at the pool limit) before it is granted again."""

    def _decode(self, margin=2.0):
        return DecodeInstance(
            "decode-0", pool_tokens=2_000, watermark=1.0,
            perf=DecodePerf(base_ms=1.0, slope_ms=0.0),
            max_running_requests=10, clip_max_new_tokens=4096,
            num_reserved_decode_tokens=0, init_new_token_ratio=0.01,
            min_new_token_ratio=0.01, ratio_decay_per_iter=0.0,
            page_size=1, retract_retry_margin=margin)

    def test_retry_needs_extra_headroom(self):
        d = self._decode()
        d.d_used = 1_000  # allocatable = 1000
        spec = RequestSpec(rid=0, input_tokens=600, output_tokens=10)
        fresh = BootstrapRequest(spec=spec, origin="p")
        retry = BootstrapRequest(spec=spec, origin="p", retracted=True,
                                 attempt=1)
        self.assertEqual(d._verdict(fresh), "accept")   # needs ~600
        self.assertEqual(d._verdict(retry), "wait")     # needs 1200

    def test_margin_capped_at_limit_so_retry_eventually_fits(self):
        d = self._decode(margin=100.0)
        spec = RequestSpec(rid=0, input_tokens=600, output_tokens=10)
        retry = BootstrapRequest(spec=spec, origin="p", retracted=True,
                                 attempt=1)
        # empty pool: the cap keeps the requirement at the pool limit
        self.assertEqual(d._verdict(retry), "accept")

    def test_stale_chunk_dropped_after_retry(self):
        d = self._decode()
        engine = Engine()
        engine.register(d)
        ctx = Context(engine, 0.0)
        spec = RequestSpec(rid=0, input_tokens=600, output_tokens=10)
        d.handle(BootstrapRequest(spec=spec, origin="p", attempt=1), ctx)
        d.handle(KvChunk(spec=spec, tokens=100, last=False, nbytes=0,
                         attempt=0), ctx)  # from the retracted attempt
        self.assertEqual(d.admitted[0].received, 0)
        d.handle(KvChunk(spec=spec, tokens=100, last=False, nbytes=0,
                         attempt=1), ctx)
        self.assertEqual(d.admitted[0].received, 100)


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
