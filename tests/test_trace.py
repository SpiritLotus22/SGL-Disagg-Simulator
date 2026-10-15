"""Trace loading tests: timestamps, ordering, validation."""

import unittest

from cli import parse_trace


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


if __name__ == "__main__":
    unittest.main()
