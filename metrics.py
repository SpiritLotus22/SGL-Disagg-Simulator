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
    RetractRetry, RetractWaste, RouteRequest,
)


class MetricsCollector(Actor):
    actor_id = "metrics"

    def __init__(self) -> None:
        self.arrival_ms: dict[int, float] = {}
        self.grant_ms: dict[int, float] = {}
        self.first_kv_ms: dict[int, float] = {}  # first KvChunk at decode
        self.ttft_ms: dict[int, float] = {}      # last KvChunk at decode
        self.done_ms: dict[int, float] = {}
        self.decisions: dict[int, str] = {}
        self.reasons: dict[int, str] = {}
        self.retracts: dict[int, int] = defaultdict(int)
        self.rework_tokens: float = 0.0   # prefill compute lost to retracts
        self.attempts: dict[int, int] = {}  # rid -> current handshake round
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
            self.attempts[msg.spec.rid] = msg.attempt
            if msg.attempt > 0:
                # re-arm: only chunks of the new attempt count from here
                self.first_kv_ms.pop(msg.spec.rid, None)
        elif isinstance(msg, KvChunk) and ev.target.startswith("decode"):
            rid = msg.spec.rid
            if msg.attempt != self.attempts.get(rid, 0):
                pass  # stale chunk from before a retraction
            else:
                if rid not in self.first_kv_ms:
                    self.first_kv_ms[rid] = now
                if msg.last:
                    # The first token is sampled at the end of the prefill
                    # forward pass and arrives with the final KV chunk, so
                    # the last chunk's arrival is the request's TTFT.
                    self.ttft_ms[rid] = now
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
        elif isinstance(message, RetractWaste):
            self.rework_tokens += message.tokens
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
        def rel(times):
            return [times[r] - self.arrival_ms[r]
                    for r in times if r in self.arrival_ms]

        waits = rel(self.grant_ms)
        first_kv = rel(self.first_kv_ms)
        # TTFT: arrival -> last KV chunk at decode (the first token is
        # sampled at the end of prefill and arrives with it)
        ttfts = rel(self.ttft_ms)
        e2e = rel(self.done_ms)
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
            # share of arrived requests that were admitted (ACCEPT)
            "accept_rate": (decision_counts.get(ACCEPT, 0)
                            / len(self.arrival_ms)) if self.arrival_ms else 0.0,
            "retracts": sum(self.retracts.values()),
            "retract_rework_tokens": self.rework_tokens,
            "completed": len(self.done_ms),
            "avg_wait_ms": statistics.fmean(waits) if waits else 0.0,
            "avg_first_kv_ms": statistics.fmean(first_kv) if first_kv else 0.0,
            "avg_ttft_ms": statistics.fmean(ttfts) if ttfts else 0.0,
            "p95_ttft_ms": self._pct(ttfts, 0.95),
            "avg_e2e_ms": statistics.fmean(e2e) if e2e else 0.0,
            "p95_e2e_ms": self._pct(e2e, 0.95),
            "peak_prefill_util": peak_p / p_pool,
            "peak_decode_util": peak_d / d_pool,
        }
