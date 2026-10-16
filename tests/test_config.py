"""Scenario config parsing: nested flattening, aliases, unknown-key warnings."""

import unittest
import warnings

from simulator import SimConfig


class TestScenarioParsing(unittest.TestCase):
    def test_nested_schema_flattens(self):
        cfg = SimConfig.from_dict({
            "decode": {"watermark": 0.8, "pool_tokens": 12345},
            "router": {"policy": "least_loaded"},
        })
        self.assertEqual(cfg.decode_kv_watermark, 0.8)
        self.assertEqual(cfg.decode_pool_tokens, 12345)
        self.assertEqual(cfg.policy, "least_loaded")

    def test_unknown_key_warns_and_is_ignored(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = SimConfig.from_dict({"decode_pool_token": 999})
        self.assertTrue(any("decode_pool_token" in str(w.message)
                            for w in caught))
        self.assertNotEqual(cfg.decode_pool_tokens, 999)

    def test_cli_only_keys_fall_into_the_same_warning_path(self):
        # seed / trace are run parameters; in a scenario they are just
        # unknown keys -- warned about and ignored, not an error.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = SimConfig.from_dict({"seed": 1, "trace": {"path": "x"}})
        self.assertEqual(cfg.seed, 42)  # untouched
        warned = [str(w.message) for w in caught]
        self.assertTrue(any("'seed'" in m for m in warned))
        self.assertTrue(any("'trace_path'" in m for m in warned))

    def test_wait_budget_keys(self):
        cfg = SimConfig.from_dict({
            "decode": {"max_bootstrap_wait_ms": 5000},
            "prefill": {"max_queue_wait_ms": 1000},
        })
        self.assertEqual(cfg.max_bootstrap_wait_ms, 5000)
        self.assertEqual(cfg.prefill_max_queue_wait_ms, 1000)

    def test_admission_scenario_produces_no_warnings(self):
        import json
        from pathlib import Path
        scenario = json.loads(Path(
            __file__).parent.parent.joinpath(
            "scenarios", "2p2d_admission.json").read_text())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = SimConfig.from_dict(scenario)
        self.assertEqual(caught, [])
        self.assertEqual(cfg.max_bootstrap_wait_ms, 5000)
        self.assertEqual(cfg.prefill_max_queue_wait_ms, 5000)

    def test_full_scenario_produces_no_warnings(self):
        import json
        from pathlib import Path
        scenario = json.loads(Path(
            __file__).parent.parent.joinpath(
            "scenarios", "2p2d.json").read_text())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SimConfig.from_dict(scenario)
        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
