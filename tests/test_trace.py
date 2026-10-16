"""Trace loading tests: timestamps, ordering, validation."""

import unittest

from cli import parse_trace, trace_hidden_types
from protocol import (
    DecodeHeartbeat, DecodeTick, KvChunk, PrefillHeartbeat, PrefillTick,
)
from simulator import SimConfig, build


class TestTrace(unittest.TestCase):
    def test_explicit_timestamps(self):
        raw = [
            {"input": 100, "output": 10, "arrival_ms": 0},
            {"input": 200, "output": 10, "arrival_ms": 2500},
            {"input": 300, "output": 10, "arrival_ms": 900},
        ]
        times = [t for t, _ in parse_trace(raw)]
        self.assertEqual(times, [0.0, 2500.0, 900.0])  # order preserved

    def test_missing_timestamp_rejected(self):
        raw = [{"input": 100, "output": 10}]
        with self.assertRaises(ValueError):
            parse_trace(raw)

    def test_missing_required_fields_rejected(self):
        with self.assertRaises(ValueError):
            parse_trace([{"arrival_ms": 0, "output": 10}])   # no input
        with self.assertRaises(ValueError):
            parse_trace([{"arrival_ms": 0, "input": 100}])   # no output

    def test_non_object_entry_rejected(self):
        with self.assertRaises(ValueError):
            parse_trace([[0, 100, 10]])

    def test_non_numeric_field_rejected(self):
        with self.assertRaises(ValueError):
            parse_trace([{"arrival_ms": "soon", "input": 100, "output": 10}])

    def test_negative_timestamp_rejected(self):
        raw = [{"input": 100, "output": 10, "arrival_ms": -5}]
        with self.assertRaises(ValueError):
            parse_trace(raw)

    def test_token_validation(self):
        with self.assertRaises(ValueError):
            parse_trace([{"input": 100, "output": 10,
                          "cache_hit": 200, "arrival_ms": 0}])
        with self.assertRaises(ValueError):
            parse_trace([{"input": -1, "output": 10, "arrival_ms": 0}])

    def test_cache_hit_carried(self):
        _, spec = parse_trace([{"input": 1000, "output": 10,
                                "cache_hit": 400, "arrival_ms": 0}])[0]
        self.assertEqual(spec.extend_len, 600)

    def test_bundled_traces_load(self):
        from pathlib import Path
        from cli import load_trace
        for name in ("example.json", "time-series.json", "heavy-mixed.json"):
            trace = load_trace(Path(__file__).parent.parent.joinpath(
                "examples", name))
            self.assertTrue(trace)


class TestTraceEventFilter(unittest.TestCase):
    """--trace output: tick/heartbeat messages are hidden unless the
    scenario's trace section re-enables them one by one."""

    def test_hidden_by_default(self):
        hidden = trace_hidden_types(SimConfig())
        self.assertEqual(
            hidden, {PrefillTick, DecodeTick, PrefillHeartbeat, DecodeHeartbeat})

    def test_scenario_reenables_one_by_one(self):
        config = SimConfig.from_dict({
            "trace": {"print_prefill_tick": True,
                      "print_prefill_heartbeat": True},
        })
        hidden = trace_hidden_types(config)
        self.assertEqual(hidden, {DecodeTick, DecodeHeartbeat})

    def test_filtered_event_stream(self):
        raw = [{"input": 1000, "output": 5, "cache_hit": 0, "arrival_ms": 0}]
        sim = build(SimConfig(), parse_trace(raw))
        hidden = trace_hidden_types(sim.config)
        seen = []
        sim.engine.add_observer(
            lambda now, ev: seen.append(ev)
            if type(ev.message) not in hidden else None)
        sim.run()
        seen_types = {type(ev.message) for ev in seen}
        self.assertNotIn(PrefillTick, seen_types)
        self.assertNotIn(PrefillHeartbeat, seen_types)
        self.assertIn(KvChunk, seen_types)  # real traffic still recorded


if __name__ == "__main__":
    unittest.main()
