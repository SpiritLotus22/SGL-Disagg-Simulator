"""Multi-instance routing: the policy picks both P and D per request."""

import unittest

from cli import parse_trace
from simulator import SimConfig, build
from protocol import AssignPrefill, KvChunk


def small_requests(n):
    return [{"input": 2000, "output": 20, "cache_hit": 0,
             "arrival_ms": i * 100} for i in range(n)]


class ObservedTargets:
    def __init__(self):
        self.assign_targets = set()
        self.chunk_targets = set()

    def __call__(self, now, ev):
        if isinstance(ev.message, AssignPrefill):
            self.assign_targets.add(ev.target)
        elif isinstance(ev.message, KvChunk):
            self.chunk_targets.add(ev.target)


class TestMultiInstance(unittest.TestCase):
    def test_round_robin_uses_all_instances(self):
        config = SimConfig(prefill_instances=2, decode_instances=2,
                           policy="round_robin")
        sim = build(config, parse_trace(small_requests(6)))
        obs = ObservedTargets()
        sim.engine.add_observer(obs)
        summary = sim.run()
        self.assertEqual(summary["completed"], 6)
        self.assertEqual(obs.assign_targets, {"prefill-0", "prefill-1"})
        self.assertEqual(obs.chunk_targets, {"decode-0", "decode-1"})

    def test_least_loaded_uses_all_instances(self):
        config = SimConfig(prefill_instances=2, decode_instances=2,
                           policy="least_loaded", seed=7)
        sim = build(config, parse_trace(small_requests(6)))
        obs = ObservedTargets()
        sim.engine.add_observer(obs)
        summary = sim.run()
        self.assertEqual(summary["completed"], 6)
        self.assertEqual(obs.assign_targets, {"prefill-0", "prefill-1"})
        self.assertEqual(obs.chunk_targets, {"decode-0", "decode-1"})

    def test_asymmetric_fleet(self):
        # 2 prefills, 1 decode: both prefills must reach the same decode
        config = SimConfig(prefill_instances=2, decode_instances=1,
                           policy="round_robin")
        sim = build(config, parse_trace(small_requests(4)))
        obs = ObservedTargets()
        sim.engine.add_observer(obs)
        summary = sim.run()
        self.assertEqual(summary["completed"], 4)
        self.assertEqual(obs.assign_targets, {"prefill-0", "prefill-1"})
        self.assertEqual(obs.chunk_targets, {"decode-0"})

    def test_determinism_with_multiple_instances(self):
        config = SimConfig(prefill_instances=2, decode_instances=2,
                           policy="least_loaded", seed=7)
        trace = parse_trace(small_requests(6))
        a = build(config, trace).run()
        b = build(config, trace).run()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
