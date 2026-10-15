# SGLang PD Disagg Simulator

PD 分离（Prefill/Decode Disaggregated）LLM 推理系统的离散事件模拟器。
输入一份带时间戳的请求序列（trace），在虚拟时间上重放整个系统的运行——
router 调度、prefill 计算、KV 跨节点传输、decode 逐轮生成、显存准入与
回撤——最后输出 TTFT、排队时长、端到端时延、双池利用率等毫秒级指标。

用途：在不动真实集群的情况下，研究水位、池容量、实例数量、调度策略对
系统行为（排队、拒绝、retract）的影响。

## 目录

- [快速开始](#快速开始)
- [构造请求序列](#构造请求序列)
- [命令行参数](#命令行参数)
- [场景配置](#场景配置)
- [工作原理](#工作原理)
  - [日程表：事件的排序与重放](#日程表事件的排序与重放)
  - [角色与消息协议](#角色与消息协议)
  - [一条请求的完整生命周期](#一条请求的完整生命周期)
  - [Prefill 建模](#prefill-建模)
  - [Decode 建模](#decode-建模)
  - [KV 传输建模](#kv-传输建模)
  - [Router 与调度策略](#router-与调度策略)
  - [指标采集](#指标采集)
  - [确定性](#确定性)
- [目录结构](#目录结构)
- [待改进](#待改进)

## 快速开始

纯标准库实现，无第三方依赖，Python 3.10+：

```bash
# 生成一条 100 请求的示例序列（examples/time-series.json）
python examples/series_generate.py

# 用默认参数回放并扫描 5 档 decode 水位
python cli.py --input examples/time-series.json

# 叠加多实例场景配置，只看 90% 水位
python cli.py --input examples/time-series.json \
    --config scenarios/two_p_two_d.json --watermarks 0.9

# 运行测试
python -m unittest discover -s tests -t . -v
```

## 构造请求序列

模拟器只接受 trace 文件：一个 JSON 数组，每条请求必须带 `arrival_ms`
（到达时刻，毫秒）：

```json
[
  { "arrival_ms": 61.2,  "input": 5722,  "output": 1046, "cache_hit": 0 },
  { "arrival_ms": 305.8, "input": 12338, "output": 388,  "cache_hit": 4000 }
]
```

字段校验：`arrival_ms`/`input`/`output` 为必填且非负；`cache_hit` 可选
（默认 0）且不得超过 `input`。`output` 的语义是 EOS 实际出现的位置；
真实生成还会被配置里的 `decode.max_new_tokens`（声明上限）截断。

`examples/series_generate.py` 是一个现成的生成器：

```bash
python series_generate.py [输出文件] [条目数量]
# 默认值：输出到脚本同目录的 time-series.json，100 条
```

生成规律：到达间隔 85% 为 50–500ms 的常规间隔、10% 为 5–30ms 的突发、
5% 为 2–5s 的空闲段；`input`/`output` 按对数正态分布采样（input 封顶
120k token）；25% 的请求带 20%–80% 的前缀命中。固定随机种子，同一命令
永远生成同一文件。

如果你想用别的流量模型（泊松、MMPP 突发）或者转换真实数据集（Azure
LLM trace、Mooncake trace 等），写一个离线脚本输出上述格式即可，不需要
改动模拟器。

## 命令行参数

```text
--input PATH        请求 trace 文件（必填）
--config PATH       场景配置文件（可选，见下节）
--watermarks F...   要扫描的 decode KV 水位，可给多个
                    （默认 0.6 0.7 0.8 0.9 0.95）
--seed N            调度策略的随机种子（默认 42）
--trace             把完整事件流写入 traces/trace_<时间戳>.txt
                    （仅支持单水位：多水位时事件流太长，无法逐一阅读）
```

输出分两种模式：单水位打印完整 JSON 汇总（各分位数齐全）；多水位打印
对比表（每档水位一行：决策计数、retract 数、双池峰值利用率、平均等待、
TTFT/E2E 的 p95）。

## 场景配置

场景文件（JSON）只描述**系统**——实例数量、池容量、速率、调度策略。
`seed` 和 trace 输入是运行参数，只能走命令行；场景里出现任何无法识别
的键（包括笔误）会打印警告并忽略，不会中断运行。完整示例：

```json
{
  "sim":      { "max_sim_time_ms": 600000 },
  "router":   { "policy": "round_robin" },
  "prefill":  { "instances": 2, "pool_tokens": 131072, "tps": 40000,
                "chunked_prefill_size": 8192 },
  "decode":   { "instances": 2, "pool_tokens": 200000, "watermark": 0.9,
                "max_running_requests": 256, "iter_base_ms": 50,
                "iter_slope_ms_per_req": 0.5, "max_new_tokens": 4096,
                "clip_max_new_tokens": 4096,
                "num_reserved_decode_tokens": 512,
                "init_new_token_ratio": 0.9, "min_new_token_ratio": 0.5 },
  "transfer": { "bandwidth_gbps": 50, "base_latency_ms": 2,
                "bytes_per_token_kb": 128 }
}
```

| 字段 | 默认值 | 含义 |
|---|---|---|
| `sim.max_sim_time_ms` | 600000 | 虚拟时钟上限，到达后截断 |
| `router.policy` | `round_robin` | 调度策略，可选 `round_robin` / `least_loaded` |
| `prefill.instances` | 1 | prefill 实例数 |
| `prefill.pool_tokens` | 131072 | 单实例 KV 池容量（token） |
| `prefill.tps` | 40000 | 单实例 prefill 吞吐（token/秒，共享） |
| `prefill.chunked_prefill_size` | 8192 | 每个 KV chunk 的 token 数；`null` 关闭切块 |
| `prefill.tick_ms` | 10 | prefill 调度 tick 的时长（算力分配粒度） |
| `decode.instances` | 1 | decode 实例数 |
| `decode.pool_tokens` | 200000 | 单实例 KV 池容量（token） |
| `decode.watermark` | 0.90 | 准入水位（池容量 × 水位 = 准入上限） |
| `decode.max_running_requests` | 256 | 单实例最大并发请求数 |
| `decode.iter_base_ms` | 50 | decode 迭代基础时延（batch=0） |
| `decode.iter_slope_ms_per_req` | 0.5 | 每多一个并发请求增加的迭代时延 |
| `decode.max_new_tokens` | 128 | 声明的输出上限：准入估算和生成截断都用它，对应 sglang `SamplingParams.max_new_tokens` 的默认值； trace 的 `output` 只是 EOS 落点，不参与估算 |
| `decode.clip_max_new_tokens` | 4096 | 乐观准入中未来生成量估算的封顶 |
| `decode.num_reserved_decode_tokens` | 512 | 每请求固定预留 token 数 |
| `decode.init_new_token_ratio` | 0.9 | 准入估算的初始悲观系数 |
| `decode.min_new_token_ratio` | 0.5 | 悲观系数下限（最乐观） |
| `decode.ratio_decay_per_iter` | 0.01 | 每轮迭代的衰减量 |
| `decode.page_size` | 16 | KV 页对齐粒度 |
| `decode.retract_retry_margin` | 2.0 | retract 重试的准入余量倍数（封顶在准入上限，空池一定能再进） |
| `decode.head_of_line_blocking` | true | bootstrap 队列是否队首阻塞 |
| `transfer.bandwidth_gbps` | 50 | KV 传输带宽（GB/s，按每 P→D 通道） |
| `transfer.base_latency_ms` | 2 | 单次传输基础时延 |
| `transfer.bytes_per_token_kb` | 128 | 每 token 的 KV 字节数 |
| `context_len` | 131072 | prompt 硬上限（顶层字段） |

`scenarios/` 下有两个现成的多实例场景：`two_p_two_d.json`（2P+2D，
round-robin）和 `four_p_two_d.json`（4P+2D，least-loaded）。

## 工作原理

### 日程表：事件的排序与重放

核心在 `engine.py`。整个模拟器就是一张**日程表**：一个最小堆，里面每
个元素是

```text
Event = (time_ms, priority, seq, target, message)
```

主循环（`Engine.run`）反复做三件事：弹出堆顶（时间最早的事件）、把
虚拟时钟 `now_ms` 跳到该时刻、把消息派发给目标角色（`Actor.handle`）。
处理过程中角色通过 `ctx.emit(target, message, delay_ms)` 把新事件插回
堆里。堆空或超过 `max_sim_time_ms` 时结束。

两个关键点：

- **排序键是三元组**：先按时间，同刻按优先级（释放显存 > 内部迭代 >
  准入判断 > 新到达，避免同刻竞态），再同则按插入序号（先进先出）。
  三者齐备才能保证同一输入永远产生同一事件序列。
- **空闲期不耗算力**：两个相邻事件之间系统状态不可能变化，时钟直接
  跳过去。一小时跨度的 trace 一秒内跑完。

`engine.py` 不认识 prefill/decode 这些词，它是通用的排队系统内核。

### 角色与消息协议

业务侧有三个角色（`actors/`），它们**只通过消息通信**——不允许直接
调用对方方法、读对方属性。所有消息类型定义在 `protocol.py`，是不可变
数据类：

| 消息 | 方向 | 含义 |
|---|---|---|
| `RouteRequest` | 装配层 → Router | 一条请求到达 |
| `AssignPrefill` | Router → P | 调度决策（携带选定的 `decode_id`） |
| `BootstrapRequest` | P → D | 请求 decode 侧预分配 KV |
| `BootstrapGrant` / `BootstrapWait` / `BootstrapReject` | D → P | 成功 / 排队（等待不是拒绝）/ 硬拒 |
| `KvChunk` | P → D | 一块 KV 数据，边算边传 |
| `RetractRetry` | D → P | 显存超支，请求回炉重做 |
| `RetractWaste` | P → metrics | 一次回撤丢弃的 prefill 算力 |
| `DoneNotice` | D → Router | 请求完成 |
| `Heartbeat` / `DecodeHeartbeat` | P/D → Router | 负载视图上报 |
| `DecisionNotice` | 任意 → metrics | 不产生其他流量的硬决策 |

这套协议的形状与 sglang 的 PD 分离实现对应：bootstrap 握手对应
`PrefillBootstrapQueue`，chunk 流式传输对应 `send_kv_chunk`。

### 一条请求的完整生命周期

1. **装配时**（`simulator.build`）：trace 里每条请求按 `arrival_ms`
   直接排入日程表；
2. **到达**：Router 做入口校验（`input > context_len` 或
   `input > prefill 池容量` → 硬拒 `REJECT_PREFILL_PRESSURE`），否则
   调度策略为这条请求**同时**选出 P 和 D 实例，发 `AssignPrefill`；
3. **bootstrap 握手**：P 向选定的 D 发 `BootstrapRequest`。D 计算
   `required = ceil_page(input) + num_reserved_decode_tokens`：
   - `required > 池容量 × 水位`（独占也放不下）→ 硬拒 `REJECT_DECODE_KV`；
   - 头寸够 → **真实预留**这部分显存，回 `BootstrapGrant`；
   - 暂时不够 → 进 D 的等待队列（backpressure，等待不是拒绝）；
4. **prefill**：P 收到 Grant 后开始计算（受 prefill 池显存闸门和共享
   算力约束），KV 边算边经传输通道发往 D；最后一个 chunk 发出后释放
   本端显存；
5. **decode**：D 收齐 KV 后开始逐轮生成，每轮每请求 1 个 token，
   轮时随并发数增长，直到 `min(output, max_new_tokens)`——`output`
   是 EOS 落点，`max_new_tokens` 是声明上限（对应 sglang 的
   FINISH_LENGTH）；释放全部显存，报
   `DoneNotice`。与 sglang 一致，**首 token 在 prefill 前向结束时
   就已采样**，随最后一块 KV 送达 D——最后一块 chunk 到达的时刻
   即该请求的 **TTFT**；
6. **retract（异常路径）**：若乐观准入放多了请求、实际占用超过物理
   池容量，D 先回撤 KV 未收齐的请求（不损失 decode 进度，只浪费在途
   prefill 算力），不够再回撤最晚开始 decode 的请求；被回撤的请求
   保留已生成 token，回 P 中止本地工作后重做握手与 prefill，且重试
   要求更大的头寸（`retract_retry_margin`）防止反复振荡；丢弃的
   prefill 算力计入 `retract_rework_tokens`，同时准入估算调回悲观。

### 准入控制链

一条请求要被系统接纳，要连过三关。三关由三个角色把守，各自只用
自己掌握的信息做判断——没有任何一关能预知真实输出长度：

1. **Router 入口校验**（`Router._route`）：只挡"永远不可能"的请求——
   `input > context_len` 或 `input > prefill 池容量`，硬拒
   `REJECT_PREFILL_PRESSURE`，不进入握手。这一关不做负载估计；
2. **Decode 侧 prealloc 闸门**（`DecodeInstance._verdict`）：在任何
   prefill 计算发生之前，先真实预留 prompt KV（`ceil_page(input) +
   num_reserved_decode_tokens`），再为已准入请求的未来生成做乐观虚拟
   预留。头寸够 → Grant；暂时不够 → 排队（等待不是拒绝）；独占也放
   不下 → 硬拒 `REJECT_DECODE_KV`；
3. **Prefill 池闸门**（`PrefillInstance._promote`）：拿到 Grant 后，
   在途 prompt 总量不得超过 `prefill.pool_tokens`，超过则排队等池
   空间。

三关全部是 backpressure 而非拒绝：除了两种硬拒，请求只会排队，不会
被丢弃。乐观估计出错时由 retract 兜底（见生命周期第 6 步）。

### Prefill 建模

`actors/prefill.py` + `perf.PrefillPerf`。

- prefill 是计算受限的：单请求独占时，耗时 ≈ `extend_len / tps`
  （`extend_len = input − cache_hit`，命中部分不占算力）；
- 多请求并发时**共享吞吐**：每个调度 tick（默认 10ms）里，本 tick 的
  算力配额（`tps × tick`）在所有活跃请求间均分；
- 显存闸门：在途 prompt 总量不得超过 `prefill.pool_tokens`，超过则
  排队等待；
- 切块传输：每攒够 `chunked_prefill_size` 个 token 就发一个 `KvChunk`；
  设为 `null` 时退化为整条算完一次性发送。

### Decode 建模

`actors/decode.py` + `perf.DecodePerf`。

- decode 是访存受限的：**每个运行中的请求每轮迭代恰好产出 1 个
  token**；
- 迭代时延随 batch 增长：`iter_latency = iter_base_ms +
  iter_slope_ms_per_req × batch_size`（默认 50ms + 0.5ms/请求）。
  吞吐 ≈ `batch / iter_latency`，饱和点自然涌现——这是 decode 侧
  backpressure 的物理来源；
- **乐观准入**：判断头寸时，除了真实占用，还为每个已准入请求虚拟
  预留 `min(max_new_tokens − 已生成, clip_max_new_tokens) ×
  new_token_ratio`——与 sglang 的 `_get_running_request_total_token_offset`
  一致，用的是声明上限而不是真实输出长度（系统无法预知 EOS）；
- **new_token_ratio 反馈环**：每轮迭代衰减（变乐观，下限
  `min_new_token_ratio`），一旦发生 retract 立刻重置回初值（变悲观）；
- **retract**：实际占用超过物理池时，先回撤 KV 未收齐的请求（最新
  grant 优先），再回撤最晚开始 decode 的请求；重试按
  `retract_retry_margin` 倍头寸准入；
- **并发上限**：`max_running_requests` 到顶后不再发放 Grant。

### KV 传输建模

`perf.TransferChannel`。每对 (P, D) 实例之间有一条逻辑通道：

- 每块 KV 的传输时延 = `base_latency_ms + 字节数 / 带宽`
  （字节数 = token 数 × `bytes_per_token_kb`）；
- 通道是**串行化**的：一块传完才传下一块——这既建模了带宽竞争，
  又保证 chunk 按序到达。

### Router 与调度策略

`actors/router.py` + `policies/`。Router 持有两侧的心跳视图（P：排队
数与在途 token 数；D：bootstrap 等待数、running 数、池占用），策略
接口为

```python
pick(spec, prefills, decodes) -> (prefill_id, decode_id)
```

**P 和 D 都是按请求当场选出的**，不存在预配对。现有两个策略：

- `policies/round_robin.py`：P、D 各自独立轮转；
- `policies/least_loaded.py`：两侧各自做 power-of-two-choices（随机抽
  两个取负载轻者，平局由种子 RNG 决胜）。

新策略加一个文件即可，不用动其他代码。

### 指标采集

`metrics.py` 的 `MetricsCollector` 以两种被动方式工作：作为引擎观察者
旁听每一次事件派发（得到到达、Grant、chunk、完成、retract 的时刻与
双池用量采样）；作为 `metrics` actor 接收 `DecisionNotice`（硬拒
决策）。它对仿真逻辑零干预。

首 token 之前的延迟拆成两个口径：

| 指标 | 口径 |
|---|---|
| `first_kv` | 到达 → 首个 KV chunk 到达 D（握手 + 首批算力 + 传输） |
| **TTFT** | 到达 → 最后一块 KV 到达 D（首 token 在 prefill 前向结束时采样，随其送达） |

汇总输出：决策计数、retract 次数与浪费的 prefill 算力
（`retract_rework_tokens`）、完成数、平均/p95 的准入等待、上述
两个口径、端到端时延、双池峰值利用率。

### 确定性

同一份 trace + 同一份配置 + 同一个 `--seed`，两次运行逐字节一致。
保证来自三点：事件三元组定序（时间、优先级、插入序号）；随机源全部
来自显式种子的 `random.Random`；不依赖任何墙钟或系统状态。有专门的
测试钉住这一点（`tests/test_retract.py::TestDeterminism`）。

## 目录结构

```text
sim_async/
├── engine.py            # 日程表内核：虚拟时钟、事件堆、派发
├── protocol.py          # 全部消息类型定义
├── perf.py              # prefill / decode / transfer 时延模型
├── metrics.py           # 指标采集
├── simulator.py         # 总装：配置 schema、连线、启动主循环
├── cli.py               # 命令行入口 + trace 加载校验
├── actors/              # router.py / prefill.py / decode.py
├── policies/            # 调度策略插件：base / round_robin / least_loaded
├── scenarios/           # 场景配置（two_p_two_d / four_p_two_d）
├── examples/            # example.json、series_generate.py、time-series.json
├── traces/              # --trace 的事件流输出
└── tests/               # 测试套件（39 个用例）
```

## 待改进

- **cache-aware 调度**：router 目前不看前缀缓存亲和性，应加入基于
  前缀匹配的 cache-aware 策略（P 侧优先复用已有缓存的实例）；
- **更多 trace 来源**：`series_generate.py` 目前只有一套分布，可以
  增加更多生成方案（泊松、MMPP 突发、昼夜 pattern），或者写转换器把
  公开数据集（Azure LLM trace、Mooncake trace 等）转成本格式；
- **异构硬件**：所有 P 实例同型号、所有 D 实例同型号；真实部署中
  P/D 常常是不同卡型，速率、池容量应支持按实例配置；
- **多级缓存命中**：`cache_hit` 是单一的输入数字，没有区分
  GPU 显存 / CPU 内存 / 远端存储的命中层级，各级命中对 prefill 时延
  和传输量的影响不同；
- **trace 格式**：当前字段名直白但冗长（`arrival_ms`/`cache_hit`），
  且缺省值支持很少，可以再做一轮美化和简化。
