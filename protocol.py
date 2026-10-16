"""Message protocol between actors, mirroring sglang's PD-disaggregation flow.

All messages are immutable dataclasses. Actors may only communicate through
these types, dispatched by the engine -- no direct method calls, no shared
mutable state between actors.
"""

from __future__ import annotations

from dataclasses import dataclass

# Decision codes.
ACCEPT = "ACCEPT"
REJECT_PREFILL_PRESSURE = "REJECT_PREFILL_PRESSURE"
REJECT_DECODE_KV = "REJECT_DECODE_KV"


@dataclass(frozen=True)
class RequestSpec:
    """Immutable description of one request."""

    rid: int
    input_tokens: int
    output_tokens: int
    cache_hit: int = 0

    @property
    def extend_len(self) -> int:
        """Uncached prompt tokens that need prefill compute."""
        return self.input_tokens - self.cache_hit


Trace = list  # list[tuple[float, RequestSpec]]: (arrival_ms, request)


# --- arrival -> router -------------------------------------------------------

@dataclass(frozen=True)
class RouteRequest:
    spec: RequestSpec
    arrival_ms: float


# --- router -> prefill -------------------------------------------------------

@dataclass(frozen=True)
class AssignPrefill:
    spec: RequestSpec
    decode_id: str  # the router picks the decode instance per request


# --- prefill <-> decode: bootstrap handshake ---------------------------------

@dataclass(frozen=True)
class BootstrapRequest:
    """Ask the decode node to pre-allocate KV before any prefill happens."""

    spec: RequestSpec
    origin: str  # prefill actor id, for the reply
    generated_so_far: int = 0  # >0 only on retract retry
    retracted: bool = False    # True when this is a retry after retraction
    attempt: int = 0           # handshake round; stale KV chunks are dropped


@dataclass(frozen=True)
class BootstrapGrant:
    """Decode side reserved KV; the request may start prefill."""

    spec: RequestSpec
    generated_so_far: int = 0
    attempt: int = 0


@dataclass(frozen=True)
class BootstrapWait:
    """Decode side has no headroom right now: queued, NOT rejected."""

    spec: RequestSpec


@dataclass(frozen=True)
class BootstrapReject:
    """The request can never fit the decode pool, even alone: hard reject."""

    spec: RequestSpec
    reason: str


# --- prefill -> decode: KV streaming -----------------------------------------

@dataclass(frozen=True)
class KvChunk:
    """One chunk of KV, streamed as it is computed (sglang send_kv_chunk)."""

    spec: RequestSpec
    tokens: int
    last: bool
    nbytes: float
    attempt: int = 0  # chunks from an older attempt arrive stale, drop them


# --- decode -> prefill / router ----------------------------------------------

@dataclass(frozen=True)
class RetractRetry:
    """Optimistic admission proved wrong: request is pushed back to prefill.
    It keeps its generated tokens and must redo bootstrap + prefill. If the
    request was still prefilling, the prefill node aborts that work."""

    spec: RequestSpec
    generated_so_far: int
    decode_id: str  # the retracting instance; the retry goes back there


@dataclass(frozen=True)
class RetractWaste:
    """Prefill compute thrown away by one retraction, for metrics."""

    rid: int
    tokens: float


@dataclass(frozen=True)
class DoneNotice:
    spec: RequestSpec


# --- prefill -> router: state view for scheduling ----------------------------

@dataclass(frozen=True)
class PrefillHeartbeat:
    """Prefill state report for the router's scheduling views."""

    prefill_id: str
    queued: int
    active_tokens: int


@dataclass(frozen=True)
class DecodeHeartbeat:
    """Decode state report for the router's scheduling views."""

    decode_id: str
    waiting: int
    running: int
    pool_used: int


# --- -> metrics --------------------------------------------------------------

@dataclass(frozen=True)
class DecisionNotice:
    """Emitted whenever a hard decision is made that produces no other
    message (e.g. router-side rejects)."""

    rid: int
    decision: str
    reason: str


# --- internal self-messages (never cross actor boundaries) -------------------

@dataclass(frozen=True)
class PrefillTick:
    pass


@dataclass(frozen=True)
class DecodeTick:
    pass
