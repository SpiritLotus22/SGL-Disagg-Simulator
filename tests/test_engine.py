"""Engine kernel tests: ordering, priorities, determinism, horizon."""

import unittest

from engine import (
    PRIORITY_ARRIVAL, PRIORITY_RELEASE, Actor, Engine,
)


class Recorder(Actor):
    def __init__(self, actor_id, log):
        self.actor_id = actor_id
        self.log = log

    def handle(self, message, ctx):
        self.log.append((ctx.now, self.actor_id, message))
        if isinstance(message, str) and message.startswith("chain"):
            ctx.emit(self.actor_id, "chain-next", delay_ms=5)


class TestEngine(unittest.TestCase):
    def test_events_in_time_order(self):
        log = []
        eng = Engine()
        eng.register(Recorder("a", log))
        eng.schedule_at(30, "a", "third")
        eng.schedule_at(10, "a", "first")
        eng.schedule_at(20, "a", "second")
        eng.run()
        self.assertEqual([m for _, _, m in log],
                         ["first", "second", "third"])

    def test_same_time_priority_then_fifo(self):
        log = []
        eng = Engine()
        eng.register(Recorder("a", log))
        eng.schedule_at(10, "a", "arrival-1", priority=PRIORITY_ARRIVAL)
        eng.schedule_at(10, "a", "release", priority=PRIORITY_RELEASE)
        eng.schedule_at(10, "a", "arrival-2", priority=PRIORITY_ARRIVAL)
        eng.run()
        self.assertEqual([m for _, _, m in log],
                         ["release", "arrival-1", "arrival-2"])

    def test_until_cutoff(self):
        log = []
        eng = Engine()
        eng.register(Recorder("a", log))
        eng.schedule_at(10, "a", "kept")
        eng.schedule_at(100, "a", "dropped")
        eng.run(until_ms=50)
        self.assertEqual([m for _, _, m in log], ["kept"])
        self.assertEqual(eng.now_ms, 10)

    def test_self_sustaining_event_chain(self):
        log = []
        eng = Engine()
        eng.register(Recorder("a", log))
        eng.schedule_at(0, "a", "chain-start")
        eng.run(until_ms=20)
        times = [t for t, _, _ in log]
        self.assertEqual(times, [0, 5, 10, 15, 20])

    def test_no_past_scheduling(self):
        eng = Engine()
        eng.register(Recorder("a", []))
        eng.now_ms = 100
        with self.assertRaises(ValueError):
            eng.schedule_at(50, "a", "too-late")


if __name__ == "__main__":
    unittest.main()
