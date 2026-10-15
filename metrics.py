"""Metrics collection: a passive observer plus a message sink.

Two channels, both strictly read-only with respect to the simulation:

  * as an engine observer it sees every dispatched event (arrivals, grants,
    chunks, completions) and derives timing metrics (TTFT, waits, E2E);
  * as a `metrics` actor it receives DecisionNotice messages for hard
    decisions that produce no other traffic (router-side rejects).

Pool/queue utilization is sampled on every event from actor references
wired up at assembly time.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from engine import Actor, Context, Event
from protocol import (
    ACCEPT, BootstrapGrant, DecisionNotice, DoneNotice, KvChunk,
    RetractRetry, RouteRequest,
)


class MetricsCollector(Actor):
    actor_id = "metrics"

    def __init__(self) -> None:
        self.arrival_ms: dict[int, float] = {}
        self.grant_ms: dict[int, float] = {}
        self.ttft_ms: dict[int, float] = {}
        self.done_ms: dict[int, float] = {}
        self.decisions: dict[int, str] = {}
        self.reasons: dict[int, str] = {}
        self.retracts: dict[int, int] = defaultdict(int)
        self.samples: list[tuple[float, int, int]] = []  # now, p_used, d_used
        self._actors = ()

    def attach(self, prefill_actors, decode_actors) -> None:
        self._actors = (tuple(prefill_actors), tuple(decode_actors))

    # -- engine observer -----------------------------------------------------

    def observe(self, now: float, ev: Event) -> None:
        msg = ev.message
        if isinstance(msg, RouteRequest):
            self.arrival_ms[msg.spec.rid] = msg.arrival_ms
        elif isinstance(msg, BootstrapGrant):
            self.grant_ms.setdefault(msg.spec.rid, now)
            self.decisions.setdefault(msg.spec.rid, ACCEPT)
        elif isinstance(msg, KvChunk) and ev.target.startswith("decode"):
            if msg.spec.rid not in self.ttft_ms:
                arr = self.arrival_ms.get(msg.spec.rid, now)
                self.ttft_ms[msg.spec.rid] = now - arr
        elif isinstance(msg, DoneNotice):
            self.done_ms[msg.spec.rid] = now
        elif isinstance(msg, RetractRetry):
            self.retracts[msg.spec.rid] += 1
        p_used = sum(p.pool_used for p in self._actors[0])
        d_used = sum(d.d_used for d in self._actors[1])
        self.samples.append((now, p_used, d_used))

    # -- actor side (DecisionNotice sink) -------------------------------------

    def handle(self, message, ctx: Context) -> None:
        if isinstance(message, DecisionNotice):
            self.decisions[message.rid] = message.decision
            self.reasons[message.rid] = message.reason
        else:
            raise TypeError(f"MetricsCollector got unexpected {type(message)}")

    # -- summary ---------------------------------------------------------------

    @staticmethod
    def _pct(values, q: float) -> float:
        if not values:
            return 0.0
        xs = sorted(values)
        idx = min(len(xs) - 1, int(q * len(xs)))
        return xs[idx]

    def summary(self) -> dict:
        waits = [self.grant_ms[r] - self.arrival_ms[r]
                 for r in self.grant_ms if r in self.arrival_ms]
        ttfts = list(self.ttft_ms.values())
        e2e = [self.done_ms[r] - self.arrival_ms[r]
               for r in self.done_ms if r in self.arrival_ms]
        decision_counts: dict[str, int] = defaultdict(int)
        for d in self.decisions.values():
            decision_counts[d] += 1
        peak_p = max((s[1] for s in self.samples), default=0)
        peak_d = max((s[2] for s in self.samples), default=0)
        p_pool = sum(p.pool_tokens for p in self._actors[0]) or 1
        d_pool = sum(d.pool_tokens for d in self._actors[1]) or 1
        return {
            "requests": len(self.arrival_ms),
            "decisions": dict(decision_counts),
            "retracts": sum(self.retracts.values()),
            "completed": len(self.done_ms),
            "avg_wait_ms": statistics.fmean(waits) if waits else 0.0,
            "avg_ttft_ms": statistics.fmean(ttfts) if ttfts else 0.0,
            "p95_ttft_ms": self._pct(ttfts, 0.95),
            "avg_e2e_ms": statistics.fmean(e2e) if e2e else 0.0,
            "p95_e2e_ms": self._pct(e2e, 0.95),
            "peak_prefill_util": peak_p / p_pool,
            "peak_decode_util": peak_d / d_pool,
        }
