"""Performance models: how fast each node is.

All parameters are meant to be calibrated from real hardware profiles and
injected via the scenario config, never hard-coded at call sites.

  prefill: compute-bound; single request runs at `tps`, concurrent requests
           share the throughput budget fairly.
  decode:  memory-bound; every running request emits exactly 1 token per
           iteration, and iteration latency grows with batch size:
               iter_latency = base_ms + slope_ms * batch_size
           This is where decode-side backpressure physically comes from.
  KV transfer: per (P, D) pair channel; chunks are serialized on the channel
           (bandwidth is shared), each chunk takes
               base_latency_ms + nbytes / bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillPerf:
    tps: float = 40_000.0              # tokens per second, shared across actives
    chunk_size: int | None = 8_192     # None disables chunking (single shot)
    tick_ms: float = 10.0              # prefill scheduling quantum

    @property
    def tokens_per_tick(self) -> float:
        return self.tps * self.tick_ms / 1000.0


@dataclass(frozen=True)
class DecodePerf:
    base_ms: float = 20.0              # iteration latency at batch size 0
    slope_ms: float = 0.5              # extra latency per concurrent request

    def iter_latency_ms(self, batch_size: int) -> float:
        return self.base_ms + self.slope_ms * batch_size


class TransferChannel:
    """One logical (P -> D) KV channel. Chunks are serialized: the channel
    finishes one transfer before starting the next, which both models
    bandwidth contention and guarantees in-order arrival."""

    def __init__(self, bandwidth_gbps: float = 50.0,
                 base_latency_ms: float = 2.0):
        # Interpret gbps as GB/s (50 GB/s ~ a 400 Gbps RDMA NIC).
        self.bytes_per_ms = bandwidth_gbps * 1e6
        self.base_latency_ms = base_latency_ms
        self.free_at = 0.0

    def delivery_delay_ms(self, now: float, nbytes: float) -> float:
        """Reserve the channel and return when the chunk arrives (relative)."""
        start = max(now, self.free_at)
        arrive = start + self.base_latency_ms + nbytes / self.bytes_per_ms
        self.free_at = arrive
        return arrive - now
