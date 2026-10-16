"""Discrete-event simulation kernel: virtual clock + event heap + actor dispatch.

This module is intentionally generic -- it knows nothing about prefill,
decode, or LLM serving. Any queueing system can be built on top of it.

Determinism contract: events are ordered by (time_ms, priority, seq), where
seq is a monotonically increasing insertion counter. Given the same scenario
and the same seeds, two runs produce byte-identical event streams.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Same-timestamp ordering: frees/decisions settle before new arrivals so that
# a request finishing at t and a request arriving at t see the post-release
# state, not the pre-release one.
PRIORITY_RELEASE = 0
PRIORITY_INTERNAL = 2
PRIORITY_DECISION = 4
PRIORITY_DEFAULT = 5
PRIORITY_ARRIVAL = 6


@dataclass(order=True)
class Event:
    time_ms: float
    priority: int
    seq: int
    target: str = field(compare=False)
    message: Any = field(compare=False)


class Context:
    """Handed to Actor.handle: the current virtual time plus emit()."""

    __slots__ = ("engine", "now")

    def __init__(self, engine: "Engine", now: float):
        self.engine = engine
        self.now = now

    def emit(self, target: str, message: Any, delay_ms: float = 0.0,
             priority: int = PRIORITY_DEFAULT) -> None:
        self.engine.schedule_at(self.now + delay_ms, target, message, priority)


class Actor:
    """Base class. Actors communicate ONLY via messages through the engine."""

    actor_id: str = ""

    def handle(self, message: Any, ctx: Context) -> None:
        raise NotImplementedError


Observer = Callable[[float, Event], None]


class Engine:
    """The schedule table: pop the earliest event, jump the clock, dispatch."""

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._seq = itertools.count()
        self.now_ms: float = 0.0
        self.actors: dict[str, Actor] = {}
        self._observers: list[Observer] = []

    def register(self, actor: Actor) -> None:
        if actor.actor_id in self.actors:
            raise ValueError(f"duplicate actor id {actor.actor_id!r}")
        self.actors[actor.actor_id] = actor

    def add_observer(self, fn: Observer) -> None:
        self._observers.append(fn)

    def schedule_at(self, time_ms: float, target: str, message: Any,
                    priority: int = PRIORITY_DEFAULT) -> None:
        if time_ms < self.now_ms - 1e-9:
            raise ValueError(
                f"cannot schedule in the past: {time_ms} < {self.now_ms}"
            )
        heapq.heappush(
            self._heap, Event(time_ms, priority, next(self._seq), target, message)
        )

    def run(self, until_ms: Optional[float] = None,
            max_events: int = 20_000_000) -> int:
        n = 0
        while self._heap:
            ev = heapq.heappop(self._heap)
            if until_ms is not None and ev.time_ms > until_ms:
                break
            self.now_ms = ev.time_ms
            for obs in self._observers:
                obs(self.now_ms, ev)
            actor = self.actors.get(ev.target)
            if actor is None:
                raise KeyError(f"no actor registered as {ev.target!r}")
            actor.handle(ev.message, Context(self, ev.time_ms))
            n += 1
            if n >= max_events:
                raise RuntimeError(f"event limit {max_events} exceeded")
        return n
