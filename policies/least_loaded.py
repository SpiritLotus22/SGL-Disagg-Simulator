"""Least-loaded: power-of-two-choices on each side independently.

Prefill side compares queue lengths; decode side compares waiting + running
requests. Uses an injected RNG so seeds stay explicit and runs stay
deterministic.
"""

from __future__ import annotations

import random

from policies.base import DecodeView, PoolView
from protocol import RequestSpec


class LeastLoaded:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def pick(self, spec: RequestSpec, prefills: list[PoolView],
             decodes: list[DecodeView]) -> tuple[str, str]:
        return (self._pick_p(prefills), self._pick_d(decodes))

    def _pick_p(self, views: list[PoolView]) -> str:
        if len(views) == 1:
            return views[0].actor_id
        a, b = self._rng.sample(views, 2)
        if a.queued == b.queued:
            return self._rng.choice([a, b]).actor_id  # break ties at random
        return (a if a.queued < b.queued else b).actor_id

    def _pick_d(self, views: list[DecodeView]) -> str:
        if len(views) == 1:
            return views[0].actor_id
        a, b = self._rng.sample(views, 2)
        load_a, load_b = a.waiting + a.running, b.waiting + b.running
        if load_a == load_b:
            return self._rng.choice([a, b]).actor_id
        return (a if load_a < load_b else b).actor_id
