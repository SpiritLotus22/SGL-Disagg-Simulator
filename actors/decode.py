"""DecodeInstance: prealloc admission + optimistic estimate + retract loop.

Semantics (on a real, continuous time axis):

  * bootstrap prealloc is the FIRST gate: KV is truly reserved on the decode
    pool before any prefill compute happens;
  * optimistic admission: future decode tokens of admitted requests are
    reserved virtually as min(max_new_tokens - generated, CLIP) *
    new_token_ratio, where max_new_tokens is the serving-wide output cap
    declared up front (mirroring SamplingParams.max_new_tokens, default
    128) -- the true output length is never known in advance;
  * new_token_ratio decays per decode iteration (more optimistic) and resets
    to its initial value on every retraction (more pessimistic);
  * if actual usage exceeds the physical pool, requests are retracted back
    to their prefill node (KV-incomplete requests first, then the most
    recently started one); they keep generated tokens and redo the
    handshake and prefill;
  * a retracted retry must clear a larger headroom (retract_retry_margin,
    capped at the pool limit) before it is granted again, so a request is
    not retracted over and over by a still-full pool;
  * waiting is backpressure, never a rejection; hard rejects only when the
    request could never fit, even alone.

Iteration rhythm comes from DecodePerf: every running request emits exactly
1 token per iteration; iteration latency grows with batch size. A request
finishes at min(output, max_new_tokens): the trace's output is where EOS
lands, and max_new_tokens is the hard cap (sglang FINISH_LENGTH).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from engine import Actor, Context, PRIORITY_INTERNAL, PRIORITY_RELEASE
from perf import DecodePerf
from protocol import (
    BootstrapGrant, BootstrapReject, BootstrapRequest, BootstrapWait,
    DecodeHeartbeat, DecodeTick, DoneNotice, KvChunk, RetractRetry,
)

ROUTER_ID = "router"


@dataclass
class _DReq:
    spec: object
    origin: str
    prealloc: int
    generated: int
    attempt: int
    grant_ms: float
    received: int = 0
    charged: int = 0        # generated tokens currently occupying the pool
    kv_complete: bool = False
    decode_start_ms: float = -1.0


class DecodeInstance(Actor):
    def __init__(self, actor_id: str, pool_tokens: int, watermark: float,
                 perf: DecodePerf, max_running_requests: int,
                 max_new_tokens: int, clip_max_new_tokens: int,
                 num_reserved_decode_tokens: int,
                 init_new_token_ratio: float, min_new_token_ratio: float,
                 ratio_decay_per_iter: float, page_size: int,
                 retract_retry_margin: float = 2.0,
                 head_of_line_blocking: bool = True):
        self.actor_id = actor_id
        self.pool_tokens = pool_tokens
        self.limit = pool_tokens * watermark
        self.perf = perf
        self.max_running = max_running_requests
        self.max_new_tokens = max_new_tokens
        self.clip = clip_max_new_tokens
        self.num_reserved = num_reserved_decode_tokens
        self.init_ratio = init_new_token_ratio
        self.min_ratio = min_new_token_ratio
        self.ratio_decay = ratio_decay_per_iter
        self.page_size = page_size
        self.retry_margin = retract_retry_margin
        self.hol_blocking = head_of_line_blocking

        self.ratio = init_new_token_ratio
        self.d_used = 0
        self.admitted: dict[int, _DReq] = {}
        self.running: list[int] = []          # rids with complete KV
        self.wait_queue: deque[BootstrapRequest] = deque()
        self._tick_scheduled = False

    # -- message handling ----------------------------------------------------

    def handle(self, message, ctx: Context) -> None:
        if isinstance(message, BootstrapRequest):
            self._on_bootstrap(message, ctx)
        elif isinstance(message, KvChunk):
            self._on_chunk(message, ctx)
        elif isinstance(message, DecodeTick):
            self._tick_scheduled = False
            self._on_tick(ctx)
        else:
            raise TypeError(f"DecodeInstance got unexpected {type(message)}")

    # -- bootstrap admission ---------------------------------------------------

    def _required(self, spec) -> int:
        pages = -(-spec.input_tokens // self.page_size) * self.page_size
        return pages + self.num_reserved

    def _offsets(self) -> float:
        total = 0.0
        for st in self.admitted.values():
            remaining = self.max_new_tokens - st.generated
            total += min(remaining, self.clip) * self.ratio
        return total

    def _verdict(self, req: BootstrapRequest) -> str:
        required = self._required(req.spec)
        if required > self.limit:
            return "never_fits"
        if len(self.admitted) >= self.max_running:
            return "wait"
        own_offset = (min(self.max_new_tokens - req.generated_so_far,
                          self.clip) * self.ratio)
        allocatable = self.limit - self.d_used - self._offsets()
        need = required + own_offset
        if req.retracted:
            # A retracted request re-entering a still-busy pool would likely
            # be retracted again, so demand extra headroom. The cap keeps an
            # empty pool able to grant any request, so retries always finish.
            need = min(required * self.retry_margin + own_offset, self.limit)
        if need <= allocatable:
            return "accept"
        return "wait"

    def _on_bootstrap(self, req: BootstrapRequest, ctx: Context) -> None:
        verdict = self._verdict(req)
        if verdict == "accept":
            self._grant(req, ctx)
        elif verdict == "wait":
            # Retries queue ahead of fresh requests: their rework is already
            # spent, so they are served first once headroom appears.
            if req.retracted:
                self.wait_queue.appendleft(req)
            else:
                self.wait_queue.append(req)
            ctx.emit(req.origin, BootstrapWait(spec=req.spec))
        else:
            required = self._required(req.spec)
            ctx.emit(req.origin, BootstrapReject(
                spec=req.spec,
                reason=f"prealloc {required} tokens exceeds decode limit "
                       f"{int(self.limit)} even alone"))
        self._heartbeat(ctx)

    def _grant(self, req: BootstrapRequest, ctx: Context) -> None:
        required = self._required(req.spec)
        self.d_used += required
        self.admitted[req.spec.rid] = _DReq(
            spec=req.spec, origin=req.origin, prealloc=required,
            generated=req.generated_so_far, attempt=req.attempt,
            grant_ms=ctx.now)
        ctx.emit(req.origin, BootstrapGrant(spec=req.spec,
                 generated_so_far=req.generated_so_far, attempt=req.attempt))

    def _rescan_wait_queue(self, ctx: Context) -> None:
        blocked = False
        pending = list(self.wait_queue)
        self.wait_queue.clear()
        for req in pending:
            if blocked:
                self.wait_queue.append(req)
                continue
            if self._verdict(req) == "accept":
                self._grant(req, ctx)
            else:
                self.wait_queue.append(req)
                blocked = self.hol_blocking

    # -- KV streaming ----------------------------------------------------------

    def _on_chunk(self, msg: KvChunk, ctx: Context) -> None:
        st = self.admitted.get(msg.spec.rid)
        if st is None or msg.attempt != st.attempt:
            return  # retracted before this chunk arrived, or re-granted since
        st.received += msg.tokens
        if msg.last and not st.kv_complete:
            st.kv_complete = True
            st.decode_start_ms = ctx.now
            self.running.append(msg.spec.rid)
            self._ensure_tick(ctx)
            self._heartbeat(ctx)

    # -- decode iterations ------------------------------------------------------

    def _ensure_tick(self, ctx: Context) -> None:
        if self.running and not self._tick_scheduled:
            self._tick_scheduled = True
            ctx.emit(self.actor_id, DecodeTick(),
                     delay_ms=self.perf.iter_latency_ms(len(self.running)),
                     priority=PRIORITY_INTERNAL)

    def _on_tick(self, ctx: Context) -> None:
        if not self.running:
            return
        # one token per running request; iteration latency priced by batch
        self.d_used += len(self.running)
        self.ratio = max(self.ratio - self.ratio_decay, self.min_ratio)
        for rid in list(self.running):
            st = self.admitted[rid]
            st.generated += 1
            st.charged += 1
            # EOS from the trace, hard-capped by the declared max_new_tokens
            if st.generated >= min(st.spec.output_tokens,
                                   self.max_new_tokens):
                self._finish(st, ctx)
        self._retract_until_fits(ctx)
        self._rescan_wait_queue(ctx)
        self._heartbeat(ctx)
        self._ensure_tick(ctx)

    def _finish(self, st: _DReq, ctx: Context) -> None:
        self.running.remove(st.spec.rid)
        del self.admitted[st.spec.rid]
        self.d_used -= st.prealloc + st.charged
        ctx.emit(ROUTER_ID, DoneNotice(spec=st.spec),
                 priority=PRIORITY_RELEASE)

    def _pick_victim(self) -> int | None:
        """Choose who to retract. Requests still waiting for their KV come
        first: freeing their prealloc costs no decode progress, only the
        prefill work in flight. Newest first in both groups."""
        inflight = [rid for rid, st in self.admitted.items()
                    if not st.kv_complete]
        if inflight:
            return max(inflight, key=lambda r: self.admitted[r].grant_ms)
        if self.running:
            return max(self.running,
                       key=lambda r: self.admitted[r].decode_start_ms)
        return None

    def _retract_until_fits(self, ctx: Context) -> None:
        retracted = False
        while self.d_used > self.pool_tokens:
            victim_rid = self._pick_victim()
            if victim_rid is None:
                break
            victim = self.admitted.pop(victim_rid)
            if victim.kv_complete:
                self.running.remove(victim_rid)
            self.d_used -= victim.prealloc + victim.charged
            ctx.emit(victim.origin, RetractRetry(
                spec=victim.spec, generated_so_far=victim.generated,
                decode_id=self.actor_id))
            retracted = True
        if retracted:
            self.ratio = self.init_ratio  # back to pessimistic

    def _heartbeat(self, ctx: Context) -> None:
        ctx.emit(ROUTER_ID, DecodeHeartbeat(
            decode_id=self.actor_id,
            waiting=len(self.wait_queue),
            running=len(self.running),
            pool_used=self.d_used,
        ))
