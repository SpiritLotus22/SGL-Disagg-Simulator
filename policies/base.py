"""Router scheduling policies: pluggable strategies for picking, per request,
BOTH a prefill instance and a decode instance. New policies are added as new
files here -- nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from protocol import RequestSpec


@dataclass(frozen=True)
class PoolView:
    """Router's (heartbeat-based) view of one prefill node."""

    actor_id: str
    queued: int
    active_tokens: int


@dataclass(frozen=True)
class DecodeView:
    """Router's (heartbeat-based) view of one decode node."""

    actor_id: str
    waiting: int
    running: int
    pool_used: int


class Policy(Protocol):
    def pick(self, spec: RequestSpec, prefills: list[PoolView],
             decodes: list[DecodeView]) -> tuple[str, str]:
        """Return (prefill_actor_id, decode_actor_id) for this request."""
        ...
