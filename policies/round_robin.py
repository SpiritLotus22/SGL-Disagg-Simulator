"""Round-robin over prefill and decode instances independently."""

from __future__ import annotations

from policies.base import DecodeView, PrefillView
from protocol import RequestSpec


class RoundRobin:
    def __init__(self) -> None:
        self._next_p = 0
        self._next_d = 0

    def pick(self, spec: RequestSpec, prefills: list[PrefillView],
             decodes: list[DecodeView]) -> tuple[str, str]:
        p = prefills[self._next_p % len(prefills)]
        d = decodes[self._next_d % len(decodes)]
        self._next_p += 1
        self._next_d += 1
        return p.actor_id, d.actor_id
