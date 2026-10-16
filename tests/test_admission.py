"""Estimated-wait admission: when the queue wait at a gate is estimated
beyond the configured budget, the request is rejected fast instead of
queueing long enough to blow up its TTFT. Retract retries are exempt.
Budgets default to None (pure backpressure, the historical behavior)."""

import unittest

from actors.decode import DecodeInstance
from cli import parse_trace
from engine import Actor, Context, Engine
from perf import DecodePerf
from protocol import (
    REJECT_DECODE_WAIT, REJECT_PREFILL_WAIT, BootstrapAbort,
    BootstrapReject, BootstrapRequest, BootstrapWait, RequestSpec,
)
from simulator import SimConfig, build


def run_trace(raw, **overrides):
    sim = build(SimConfig(**overrides), parse_trace(raw))
    return sim.run(), sim


class Recorder(Actor):
    def __init__(self, actor_id):
        self.actor_id = actor_id
        self.messages = []

    def handle(self, message, ctx):
        self.messages.append(message)


class TestDecodeWaitBudget(unittest.TestCase):
    # The decode pool fits exactly one 8k prompt; the second request must
    # wait for the first to finish.
    CONFIG = dict(
        decode_pool_tokens=10_000, decode_kv_watermark=1.0,
        num_reserved_decode_tokens=0, page_size=16,
        clip_max_new_tokens=4096, init_new_token_ratio=0.9,
        min_new_token_ratio=0.9,
        decode_iter_base_ms=10.0, decode_iter_slope_ms=0.0,
    )
    TRACE = [
        {"input": 8000, "output": 30, "cache_hit": 0, "arrival_ms": 0},
        {"input": 8000, "output": 30, "cache_hit": 0, "arrival_ms": 0},
    ]

    def test_over_budget_is_rejected(self):
        # The first request's estimated release (128 x 0.9 tokens at 10ms
        # per iteration = 1152ms) is beyond the 500ms budget, so nothing
        # is releasable in time and the second request fails fast.
        summary, _ = run_trace(self.TRACE, max_bootstrap_wait_ms=500.0,
                               **self.CONFIG)
        self.assertEqual(summary["decisions"].get("ACCEPT"), 1)
        self.assertEqual(summary["decisions"].get(REJECT_DECODE_WAIT), 1)
        self.assertEqual(summary["completed"], 1)

    def test_within_budget_queues_and_completes(self):
        summary, _ = run_trace(self.TRACE, max_bootstrap_wait_ms=2000.0,
                               **self.CONFIG)
        self.assertEqual(summary["completed"], 2)
        self.assertNotIn(REJECT_DECODE_WAIT, summary["decisions"])

    def test_default_off_preserves_backpressure(self):
        summary, _ = run_trace(self.TRACE, **self.CONFIG)
        self.assertEqual(summary["completed"], 2)
        self.assertNotIn(REJECT_DECODE_WAIT, summary["decisions"])

    def test_retry_exempt_from_budget(self):
        d = DecodeInstance(
            "decode-0", pool_tokens=2_000, watermark=1.0,
            perf=DecodePerf(base_ms=1.0, slope_ms=0.0),
            max_running_requests=10, max_new_tokens=128,
            clip_max_new_tokens=4096, num_reserved_decode_tokens=0,
            init_new_token_ratio=0.01, min_new_token_ratio=0.01,
            ratio_decay_per_iter=0.0, page_size=1,
            max_bootstrap_wait_ms=0.001)
        d.d_used = 1_900  # nearly full; no admitted request can release
        engine = Engine()
        rec, router = Recorder("p"), Recorder("router")
        engine.register(d)
        engine.register(rec)
        engine.register(router)
        ctx = Context(engine, 0.0)
        retry = BootstrapRequest(
            spec=RequestSpec(rid=0, input_tokens=600, output_tokens=10),
            origin="p", retracted=True, attempt=1)
        fresh = BootstrapRequest(
            spec=RequestSpec(rid=1, input_tokens=600, output_tokens=10),
            origin="p")
        d.handle(retry, ctx)
        d.handle(fresh, ctx)
        engine.run()
        kinds = {m.spec.rid: type(m) for m in rec.messages
                 if isinstance(m, (BootstrapWait, BootstrapReject))}
        self.assertIs(kinds[0], BootstrapWait)    # retry queued, not rejected
        self.assertIs(kinds[1], BootstrapReject)  # fresh one fails fast


class TestOverloadAB(unittest.TestCase):
    """Same trace, budget on vs off: rejection trades the unbounded TTFT
    tail for explicit failures."""

    def test_budget_bounds_ttft(self):
        raw = [{"input": 60000, "output": 2000, "cache_hit": 0,
                "arrival_ms": i * 50.0} for i in range(12)]
        off, _ = run_trace(raw, max_new_tokens=4096)
        on, _ = run_trace(raw, max_new_tokens=4096,
                          max_bootstrap_wait_ms=10_000.0)
        rejected = on["decisions"].get(REJECT_DECODE_WAIT, 0)
        self.assertGreater(rejected, 0)
        self.assertEqual(on["completed"] + rejected, 12)
        self.assertLess(on["p95_ttft_ms"], off["p95_ttft_ms"])
        self.assertLess(on["p95_ttft_ms"], 15_000)


class TestPrefillWaitBudget(unittest.TestCase):
    # The prefill pool fits exactly one 4k prompt; the second granted
    # request must wait for the first to finish prefilling (4000 tokens
    # at 40 tok/ms = 100ms).
    CONFIG = dict(
        prefill_pool_tokens=5_000,
        decode_pool_tokens=200_000,  # decode side never blocks here
        max_new_tokens=4096,
    )
    TRACE = [
        {"input": 4000, "output": 30, "cache_hit": 0, "arrival_ms": 0},
        {"input": 4000, "output": 30, "cache_hit": 0, "arrival_ms": 0},
    ]

    def test_over_budget_releases_decode_prealloc(self):
        summary, sim = run_trace(self.TRACE, prefill_max_queue_wait_ms=50.0,
                                 **self.CONFIG)
        self.assertEqual(summary["decisions"].get(REJECT_PREFILL_WAIT), 1)
        self.assertEqual(summary["completed"], 1)
        decode = sim.engine.actors["decode-0"]
        prefill = sim.engine.actors["prefill-0"]
        self.assertEqual(decode.d_used, 0)     # prealloc released
        self.assertEqual(len(prefill.waiting), 0)

    def test_within_budget_completes(self):
        summary, _ = run_trace(self.TRACE, prefill_max_queue_wait_ms=500.0,
                               **self.CONFIG)
        self.assertEqual(summary["completed"], 2)
        self.assertNotIn(REJECT_PREFILL_WAIT, summary["decisions"])

    def test_default_off_preserves_backpressure(self):
        summary, _ = run_trace(self.TRACE, **self.CONFIG)
        self.assertEqual(summary["completed"], 2)

    def test_abort_after_retract_is_ignored(self):
        # A retract racing with the abort: the admitted entry is already
        # gone, the abort must not double-free the pool.
        d = DecodeInstance(
            "decode-0", pool_tokens=2_000, watermark=1.0,
            perf=DecodePerf(base_ms=1.0, slope_ms=0.0),
            max_running_requests=10, max_new_tokens=128,
            clip_max_new_tokens=4096, num_reserved_decode_tokens=0,
            init_new_token_ratio=0.01, min_new_token_ratio=0.01,
            ratio_decay_per_iter=0.0, page_size=1)
        engine = Engine()
        router = Recorder("router")
        engine.register(d)
        engine.register(Recorder("p"))
        engine.register(router)
        ctx = Context(engine, 0.0)
        spec = RequestSpec(rid=0, input_tokens=600, output_tokens=10)
        d.handle(BootstrapRequest(spec=spec, origin="p"), ctx)
        self.assertIn(0, d.admitted)
        d.d_used = d.pool_tokens + 1  # force an overrun
        d._retract_until_fits(ctx)    # pops the request, emits RetractRetry
        used_after_retract = d.d_used
        d.handle(BootstrapAbort(spec=spec, attempt=0), ctx)
        self.assertEqual(d.d_used, used_after_retract)
        self.assertNotIn(0, d.admitted)


class TestWaitEstimate(unittest.TestCase):
    """Adaptive output-length estimate: fixed default until the completion
    window fills, then the window's p90, capped by clip/max_new_tokens."""

    def _decode(self):
        return DecodeInstance(
            "decode-0", pool_tokens=100_000, watermark=1.0,
            perf=DecodePerf(base_ms=1.0, slope_ms=0.0),
            max_running_requests=100, max_new_tokens=4096,
            clip_max_new_tokens=4096, num_reserved_decode_tokens=0,
            init_new_token_ratio=0.9, min_new_token_ratio=0.5,
            ratio_decay_per_iter=0.01, page_size=1,
            wait_est_default_tokens=640, wait_est_window=100)

    def test_default_before_window_fills(self):
        d = self._decode()
        d._output_window.extend([10] * 99)
        self.assertEqual(d._est_output_tokens(), 640)

    def test_p90_after_window_fills(self):
        d = self._decode()
        d._output_window.extend([100] * 90 + [2000] * 10)
        # sorted[90] is the first element of the top decile
        self.assertEqual(d._est_output_tokens(), 2000)

    def test_capped_by_clip(self):
        d = self._decode()
        d._output_window.extend([100_000] * 100)
        self.assertEqual(d._est_output_tokens(), 4096)

    def test_capped_by_max_new_tokens(self):
        d = self._decode()
        d.max_new_tokens = 128
        self.assertEqual(d._est_output_tokens(), 128)


if __name__ == "__main__":
    unittest.main()
