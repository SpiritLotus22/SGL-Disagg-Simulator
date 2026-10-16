"""Assembly: build a wired-up simulation from a config + a trace, then run.

SimConfig mirrors the scenario JSON schema (see scenarios/). The builder
creates the engine, all actors, transfer channels, and the metrics
collector, connects them, and schedules every arrival from the trace.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Optional

from actors.decode import DecodeInstance
from actors.prefill import PrefillInstance
from actors.router import Router
from engine import Engine, PRIORITY_ARRIVAL
from metrics import MetricsCollector
from perf import DecodePerf, PrefillPerf, TransferChannel
from protocol import RouteRequest, Trace


@dataclass(frozen=True)
class SimConfig:
    seed: int = 42
    max_sim_time_ms: float = 600_000.0
    context_len: int = 131_072

    # router
    policy: str = "round_robin"

    # prefill instances
    prefill_instances: int = 1
    prefill_pool_tokens: int = 131_072
    prefill_tps: float = 40_000.0
    chunked_prefill_size: Optional[int] = 8_192
    prefill_tick_ms: float = 10.0

    # decode instances
    decode_instances: int = 1
    decode_pool_tokens: int = 200_000
    decode_kv_watermark: float = 0.90
    max_running_requests: int = 256
    decode_iter_base_ms: float = 50.0
    decode_iter_slope_ms: float = 0.5
    max_new_tokens: int = 128
    clip_max_new_tokens: int = 4_096
    num_reserved_decode_tokens: int = 512
    init_new_token_ratio: float = 0.9
    min_new_token_ratio: float = 0.5
    ratio_decay_per_iter: float = 0.01
    page_size: int = 16
    retract_retry_margin: float = 2.0
    head_of_line_blocking: bool = True

    # KV transfer
    transfer_bandwidth_gbps: float = 50.0
    transfer_base_latency_ms: float = 2.0
    bytes_per_token_kb: float = 128.0

    # --trace event output: tick/heartbeat messages flood the stream, so
    # each kind needs an explicit opt-in via the scenario's trace section
    trace_print_prefill_tick: bool = False
    trace_print_decode_tick: bool = False
    trace_print_heartbeat: bool = False
    trace_print_decode_heartbeat: bool = False

    @classmethod
    def from_dict(cls, cfg: dict) -> "SimConfig":
        """Build a config from a nested scenario dict.

        Nested sections are flattened ("decode": {"watermark": x} becomes
        decode_watermark) and mapped through the alias table. Any key that
        does not resolve to a known scenario field is ignored with a
        warning, so a typo can never silently change a run.
        """
        flat: dict = {}
        for key, value in cfg.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat[f"{key}_{sub_key}"] = sub_val
            else:
                flat[key] = value
        # Canonical scenario keys -> SimConfig fields. A key not listed here
        # (typo, stale field, CLI-only run parameter like seed) is reported
        # and ignored via the same warning path.
        alias = {
            "sim_max_sim_time_ms": "max_sim_time_ms",
            "router_policy": "policy",
            "prefill_instances": "prefill_instances",
            "prefill_pool_tokens": "prefill_pool_tokens",
            "prefill_tps": "prefill_tps",
            "prefill_chunked_prefill_size": "chunked_prefill_size",
            "prefill_tick_ms": "prefill_tick_ms",
            "decode_instances": "decode_instances",
            "decode_pool_tokens": "decode_pool_tokens",
            "decode_watermark": "decode_kv_watermark",
            "decode_max_running_requests": "max_running_requests",
            "decode_iter_base_ms": "decode_iter_base_ms",
            "decode_iter_slope_ms_per_req": "decode_iter_slope_ms",
            "decode_max_new_tokens": "max_new_tokens",
            "decode_clip_max_new_tokens": "clip_max_new_tokens",
            "decode_num_reserved_decode_tokens": "num_reserved_decode_tokens",
            "decode_init_new_token_ratio": "init_new_token_ratio",
            "decode_min_new_token_ratio": "min_new_token_ratio",
            "decode_ratio_decay_per_iter": "ratio_decay_per_iter",
            "decode_page_size": "page_size",
            "decode_retract_retry_margin": "retract_retry_margin",
            "decode_head_of_line_blocking": "head_of_line_blocking",
            "transfer_bandwidth_gbps": "transfer_bandwidth_gbps",
            "transfer_base_latency_ms": "transfer_base_latency_ms",
            "transfer_bytes_per_token_kb": "bytes_per_token_kb",
            "context_len": "context_len",
            "trace_print_prefill_tick": "trace_print_prefill_tick",
            "trace_print_decode_tick": "trace_print_decode_tick",
            "trace_print_heartbeat": "trace_print_heartbeat",
            "trace_print_decode_heartbeat": "trace_print_decode_heartbeat",
        }
        kwargs = {}
        for k, v in flat.items():
            if k in alias:
                kwargs[alias[k]] = v
            else:
                warnings.warn(
                    f"sim_async: ignoring unexpected scenario parameter {k!r}"
                )
        return cls(**kwargs)


def make_policy(name: str, seed: int):
    if name == "round_robin":
        from policies.round_robin import RoundRobin
        return RoundRobin()
    if name == "least_loaded":
        from policies.least_loaded import LeastLoaded
        return LeastLoaded(random.Random(seed))
    raise ValueError(
        f"unknown policy {name!r} (cache_aware lands in Phase 2)"
    )


@dataclass
class Simulation:
    engine: Engine
    collector: MetricsCollector
    config: SimConfig

    def run(self) -> dict:
        self.engine.run(until_ms=self.config.max_sim_time_ms)
        return self.collector.summary()


def build(config: SimConfig, trace: Trace) -> Simulation:
    engine = Engine()
    collector = MetricsCollector()

    prefill_ids = [f"prefill-{i}" for i in range(config.prefill_instances)]
    decode_ids = [f"decode-{i}" for i in range(config.decode_instances)]
    bytes_per_token = config.bytes_per_token_kb * 1024.0

    p_perf = PrefillPerf(
        tps=config.prefill_tps,
        chunk_size=config.chunked_prefill_size,
        tick_ms=config.prefill_tick_ms,
    )
    d_perf = DecodePerf(
        base_ms=config.decode_iter_base_ms,
        slope_ms=config.decode_iter_slope_ms,
    )

    prefills, decodes = [], []
    for pid in prefill_ids:
        # one transfer channel per (P, D) pair; the router picks the pair
        # per request, so channels are not bound to a single peer
        channels = {did: TransferChannel(config.transfer_bandwidth_gbps,
                                         config.transfer_base_latency_ms)
                    for did in decode_ids}
        prefills.append(PrefillInstance(pid, channels,
                                        config.prefill_pool_tokens,
                                        p_perf, bytes_per_token))
    for did in decode_ids:
        decodes.append(DecodeInstance(
            did, config.decode_pool_tokens, config.decode_kv_watermark,
            d_perf, config.max_running_requests, config.max_new_tokens,
            config.clip_max_new_tokens,
            config.num_reserved_decode_tokens, config.init_new_token_ratio,
            config.min_new_token_ratio, config.ratio_decay_per_iter,
            config.page_size, config.retract_retry_margin,
            config.head_of_line_blocking))

    router = Router(config.context_len, config.prefill_pool_tokens,
                    prefill_ids, decode_ids,
                    make_policy(config.policy, config.seed))

    for actor in (*prefills, *decodes, router, collector):
        engine.register(actor)
    collector.attach(prefills, decodes)
    engine.add_observer(collector.observe)
    # A trace is static: schedule every arrival directly onto the calendar.
    for time_ms, spec in trace:
        engine.schedule_at(time_ms, router.actor_id,
                           RouteRequest(spec=spec, arrival_ms=time_ms),
                           priority=PRIORITY_ARRIVAL)
    return Simulation(engine, collector, config)
