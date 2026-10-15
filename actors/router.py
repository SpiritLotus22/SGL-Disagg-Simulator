"""Router: entry validation + scheduling policy + forwarding.

Hard rejects (prompt > context_len or > prefill pool) happen here and never
enter the bootstrap handshake. For every request the policy picks BOTH a
prefill instance and a decode instance, using heartbeat-based views of both
sides.
"""

from __future__ import annotations

from engine import Actor, Context
from policies.base import DecodeView, Policy, PoolView
from protocol import (
    REJECT_PREFILL_PRESSURE, AssignPrefill, DecisionNotice, DoneNotice,
    DecodeHeartbeat, Heartbeat, RouteRequest,
)

METRICS_ID = "metrics"


class Router(Actor):
    actor_id = "router"

    def __init__(self, context_len: int, prefill_pool_tokens: int,
                 prefill_ids: list[str], decode_ids: list[str],
                 policy: Policy):
        self.context_len = context_len
        self.prefill_pool_tokens = prefill_pool_tokens
        self.prefill_ids = sorted(prefill_ids)
        self.decode_ids = sorted(decode_ids)
        self.policy = policy
        self.prefill_views: dict[str, PoolView] = {
            pid: PoolView(pid, 0, 0) for pid in self.prefill_ids
        }
        self.decode_views: dict[str, DecodeView] = {
            did: DecodeView(did, 0, 0, 0) for did in self.decode_ids
        }

    def handle(self, message, ctx: Context) -> None:
        if isinstance(message, RouteRequest):
            self._route(message.spec, ctx)
        elif isinstance(message, Heartbeat):
            self.prefill_views[message.prefill_id] = PoolView(
                message.prefill_id, message.queued, message.active_tokens
            )
        elif isinstance(message, DecodeHeartbeat):
            self.decode_views[message.decode_id] = DecodeView(
                message.decode_id, message.waiting, message.running,
                message.pool_used,
            )
        elif isinstance(message, DoneNotice):
            pass  # completion does not change routing decisions today
        else:
            raise TypeError(f"Router got unexpected {type(message)}")

    def _route(self, spec, ctx: Context) -> None:
        if spec.input_tokens > self.context_len:
            self._reject(spec, ctx,
                         f"prompt {spec.input_tokens} exceeds context_len "
                         f"{self.context_len}")
            return
        if spec.input_tokens > self.prefill_pool_tokens:
            self._reject(spec, ctx,
                         f"prompt {spec.input_tokens} exceeds prefill pool "
                         f"{self.prefill_pool_tokens}")
            return
        pid, did = self.policy.pick(
            spec, list(self.prefill_views.values()),
            list(self.decode_views.values()))
        ctx.emit(pid, AssignPrefill(spec=spec, decode_id=did))

    def _reject(self, spec, ctx: Context, reason: str) -> None:
        ctx.emit(METRICS_ID, DecisionNotice(
            rid=spec.rid, decision=REJECT_PREFILL_PRESSURE, reason=reason,
        ))
