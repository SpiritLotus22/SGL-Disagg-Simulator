"""PrefillInstance: bootstrap initiator + shared-compute scheduler + KV sender.

The decode instance is chosen per request by the router and arrives inside
AssignPrefill; this node keeps one TransferChannel per decode peer and a
pending map (rid -> decode_id) for requests in the bootstrap handshake.

Lifecycle of one request here:

  AssignPrefill        -> send BootstrapRequest to the chosen decode node
  BootstrapGrant       -> enqueue; once the prefill pool has room, start
                          computing; KV chunks stream to the decode side as
                          they are produced; last chunk frees the local pool
  BootstrapWait        -> decode side is full; nothing to do (the decode
                          node holds the queue and will grant later)
  BootstrapReject      -> hard reject, forward the decision to metrics
  RetractRetry         -> decode pushed the request back; abort any prefill
                          work still in flight, then redo the handshake with
                          the same decode node (generated tokens kept)

Compute model: every tick (PrefillPerf.tick_ms), the throughput budget
(tps * tick) is shared fairly across active requests.
"""

from __future__ import annotations

from engine import Actor, Context, PRIORITY_INTERNAL
from perf import PrefillPerf, TransferChannel
from protocol import (
    REJECT_PREFILL_WAIT, AssignPrefill, BootstrapAbort, BootstrapGrant,
    BootstrapReject, BootstrapRequest, BootstrapWait, DecisionNotice,
    KvChunk, PrefillHeartbeat, PrefillTick, RetractRetry, RetractWaste,
)

METRICS_ID = "metrics"
ROUTER_ID = "router"


class _PReq:
    __slots__ = ("spec", "decode_id", "generated_so_far", "prefilled", "sent",
                 "attempt")

    def __init__(self, spec, decode_id: str, generated_so_far: int = 0,
                 attempt: int = 0):
        self.spec = spec
        self.decode_id = decode_id
        self.generated_so_far = generated_so_far
        self.attempt = attempt
        self.prefilled = 0.0   # computed (uncached) tokens
        self.sent = 0.0        # KV tokens already shipped (cached + computed)


class PrefillInstance(Actor):
    def __init__(self, actor_id: str, channels: dict[str, TransferChannel],
                 pool_tokens: int, perf: PrefillPerf,
                 bytes_per_token: float,
                 max_queue_wait_ms: float | None = None):
        self.actor_id = actor_id
        self.channels = channels          # one per decode instance
        self.pool_tokens = pool_tokens
        self.perf = perf
        self.bytes_per_token = bytes_per_token
        self.max_queue_wait_ms = max_queue_wait_ms
        self.pending: dict[int, str] = {}  # rid -> decode_id, in bootstrap
        self.attempts: dict[int, int] = {}  # rid -> handshake round
        self.waiting: list[_PReq] = []     # granted, awaiting pool room
        self.active: list[_PReq] = []
        self.pool_used = 0
        self._tick_scheduled = False

    # -- message handling ----------------------------------------------------

    def handle(self, message, ctx: Context) -> None:
        if isinstance(message, AssignPrefill):
            rid = message.spec.rid
            self.pending[rid] = message.decode_id
            ctx.emit(message.decode_id, BootstrapRequest(
                spec=message.spec, origin=self.actor_id,
                attempt=self.attempts.get(rid, 0)))
        elif isinstance(message, BootstrapGrant):
            decode_id = self.pending.pop(message.spec.rid)
            req = _PReq(message.spec, decode_id, message.generated_so_far,
                        message.attempt)
            if self._wait_over_budget(req):
                # Fast fail: release the decode prealloc and report the
                # rejection instead of queueing past the wait budget.
                ctx.emit(decode_id, BootstrapAbort(spec=message.spec,
                                                   attempt=message.attempt))
                ctx.emit(METRICS_ID, DecisionNotice(
                    rid=message.spec.rid, decision=REJECT_PREFILL_WAIT,
                    reason=f"estimated prefill pool wait exceeds budget "
                           f"{self.max_queue_wait_ms} ms"))
            else:
                self.waiting.append(req)
                self._promote(ctx)
        elif isinstance(message, BootstrapWait):
            pass  # backpressure: the decode node queues us; just wait
        elif isinstance(message, BootstrapReject):
            self.pending.pop(message.spec.rid, None)
            ctx.emit(METRICS_ID, DecisionNotice(
                rid=message.spec.rid, decision=message.decision,
                reason=message.reason))
        elif isinstance(message, RetractRetry):
            self._on_retract(message, ctx)
        elif isinstance(message, PrefillTick):
            self._tick_scheduled = False
            self._on_tick(ctx)
        else:
            raise TypeError(f"PrefillInstance got unexpected {type(message)}")

    # -- retraction ------------------------------------------------------------

    def _drop(self, rid: int):
        """Remove a retracted request from the local queues, if present."""
        for i, st in enumerate(self.waiting):
            if st.spec.rid == rid:
                return self.waiting.pop(i)
        for i, st in enumerate(self.active):
            if st.spec.rid == rid:
                self.pool_used -= st.spec.input_tokens
                return self.active.pop(i)
        return None

    def _on_retract(self, message: RetractRetry, ctx: Context) -> None:
        """Abort any local work, account the waste, redo the handshake."""
        rid = message.spec.rid
        st = self._drop(rid)
        # Work thrown away: partial compute if still prefilling, otherwise
        # the whole extend (finished or untouched, it will be recomputed).
        waste = st.prefilled if st is not None else message.spec.extend_len
        if waste > 0:
            ctx.emit(METRICS_ID, RetractWaste(rid=rid, tokens=waste))
        attempt = self.attempts.get(rid, 0) + 1
        self.attempts[rid] = attempt
        self.pending[rid] = message.decode_id
        ctx.emit(message.decode_id, BootstrapRequest(
            spec=message.spec, origin=self.actor_id,
            generated_so_far=message.generated_so_far,
            retracted=True, attempt=attempt))
        if st is not None:
            self._promote(ctx)  # the dropped request freed pool room

    # -- prefill scheduling --------------------------------------------------

    def _wait_over_budget(self, req: _PReq) -> bool:
        """Whether the estimated wait for pool room exceeds
        max_queue_wait_ms.

        The wait stays within budget iff the pool space releasable before
        now+budget covers this request plus everyone ahead in the waiting
        list (strict FIFO, matching _promote). Fair sharing gives every
        active the same token rate, so actives finish in ascending order
        of remaining compute and each completion speeds up the rest; only
        current actives are counted -- compute not yet admitted would make
        the estimate worse, so this is optimistic. Retries are exempt.
        """
        budget = self.max_queue_wait_ms
        if budget is None or req.attempt > 0:
            return False
        if not self.waiting and \
                self.pool_used + req.spec.input_tokens <= self.pool_tokens:
            return False  # promotes immediately, no wait at all
        need = req.spec.input_tokens
        need += sum(st.spec.input_tokens for st in self.waiting)
        available = self.pool_tokens - self.pool_used
        actives = sorted(self.active,
                         key=lambda st: st.spec.extend_len - st.prefilled)
        tokens_per_ms = self.perf.tps / 1000.0
        elapsed = 0.0
        prev = 0.0
        n = len(actives)
        for i, st in enumerate(actives):
            rem = st.spec.extend_len - st.prefilled
            elapsed += (rem - prev) * (n - i) / tokens_per_ms
            prev = rem
            if elapsed > budget:
                break
            available += st.spec.input_tokens
        return need > available

    def _promote(self, ctx: Context) -> None:
        """Move granted requests into the active set while the pool has room."""
        while self.waiting:
            st = self.waiting[0]
            if self.pool_used + st.spec.input_tokens > self.pool_tokens:
                break  # pool full -> backpressure, wait
            self.waiting.pop(0)
            self.active.append(st)
            if st.spec.extend_len > 0:
                self.pool_used += st.spec.input_tokens
            else:
                # Fully cache-hit: nothing to compute; stream cached KV now.
                self._send_chunk(ctx, st, st.spec.input_tokens, last=True)
                self.active.remove(st)
        self._heartbeat(ctx)
        self._ensure_tick(ctx)

    def _on_tick(self, ctx: Context) -> None:
        if not self.active:
            return
        share = self.perf.tokens_per_tick / len(self.active)
        for st in list(self.active):
            extend = st.spec.extend_len
            adv = min(share, extend - st.prefilled)
            st.prefilled += adv
            self._flush(ctx, st)
            if st.prefilled >= extend:
                self.active.remove(st)
                self.pool_used -= st.spec.input_tokens
        self._promote(ctx)  # freed pool room may admit waiting requests
        self._ensure_tick(ctx)

    def _flush(self, ctx: Context, st: _PReq) -> None:
        """Ship whatever KV is available, honoring the chunk size."""
        chunk = self.perf.chunk_size
        available = st.spec.cache_hit + st.prefilled - st.sent
        finished = st.prefilled >= st.spec.extend_len
        if chunk is None:
            if finished and st.sent < st.spec.input_tokens:
                self._send_chunk(ctx, st, st.spec.input_tokens - st.sent,
                                 last=True)
            return
        while available >= chunk:
            self._send_chunk(ctx, st, chunk, last=False)
            available -= chunk
        if finished and st.sent < st.spec.input_tokens:
            self._send_chunk(ctx, st, st.spec.input_tokens - st.sent,
                             last=True)

    def _send_chunk(self, ctx: Context, st: _PReq, tokens: float,
                    last: bool) -> None:
        tokens = int(tokens)
        nbytes = tokens * self.bytes_per_token
        delay = self.channels[st.decode_id].delivery_delay_ms(ctx.now, nbytes)
        st.sent += tokens
        ctx.emit(st.decode_id, KvChunk(spec=st.spec, tokens=tokens, last=last,
                                       nbytes=nbytes, attempt=st.attempt),
                 delay_ms=delay)

    # -- helpers --------------------------------------------------------------

    def _ensure_tick(self, ctx: Context) -> None:
        if self.active and not self._tick_scheduled:
            self._tick_scheduled = True
            ctx.emit(self.actor_id, PrefillTick(),
                     delay_ms=self.perf.tick_ms, priority=PRIORITY_INTERNAL)

    def _heartbeat(self, ctx: Context) -> None:
        ctx.emit(ROUTER_ID, PrefillHeartbeat(
            prefill_id=self.actor_id,
            queued=len(self.waiting),
            active_tokens=self.pool_used,
        ))
