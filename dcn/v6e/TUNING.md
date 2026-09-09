# TPU 跨 slice all-reduce 调优结论

平台 tpu7x（`tpu7x-standard-4t` × 2，DP=2，每 slice 8 device）。
负载 `dcn/benchmark.py`（SHA256 未改动，通过 runpy 加载）。
dim 32768 = 每 device 1 GiB bf16，warmups 200，best-of。

---

## 结论

两个现成 flag，不需要改图，不需要定制二进制：

```bash
export LIBTPU_INIT_ARGS="--xla_tpu_use_megascale_host_reduction=false \
                         --megascale_max_reduction_shard_size=16777216"
```

`jax.lax.psum` 跨 slice 吞吐 **162 → 331 Gbps，+104%**。

如果负载中还有基于 `ppermute` 的 DCN 通信（例如 pipeline parallel 的 P2P），
再加第三个 flag：

```bash
                         --megascale_chunk_size=16777216
```

它把 ONE_TO_ONE 路径从 273 提到 334 Gbps（+22%），代价是 `psum` 从 331 回落到 325
（落在噪声内）。只跑数据并行的话不需要它。

---

## 数据

### 第一级：关闭 host 归约

stock `libtpu 0.0.44`，3 轮交替，同进程内 `exchange_add` 作对照：

```
                       psum                            exchange_add（对照）
默认           161.1  159.1  166.1  → 162.1 ± 3.6      287.8 ± 10.9
flag=false     245.5  242.9  260.8  → 249.7 ± 9.6      277.5 ± 15.0
                                       +54.0%            −3.6%
```

Welch t ≈ 14.8。对照组不动，排除整机漂移。

### 第二级：提高归约分片上限

在 `host_reduction=false` 基础上，单 pod 内 4 配置 × 3 轮交替：

```
配置                psum            exchange_add
base            256.8 ± 12.9      273.1 ± 15.4
mrss=16Mi       331.3 ±  5.9      298.6 ±  5.7      psum +29.0%
mrss=32Mi       330.4 ±  1.7      272.8 ± 11.4
mrss=16Mi
  +chunk=16Mi   324.6 ±  8.7      334.3 ± 10.0      exch +22.4%
```

Welch（psum，base vs mrss16）：t = 9.1，p ≈ 0.002。

16 MiB 已经吃满，32 MiB 无额外收益。

`mrss=16Mi` 那一格的 `exchange_add` 298.6 不是真实效应：`mrss=32Mi` 的
`exchange_add` 是 272.8，与 base 一致。`max_reduction_shard_size` 不作用于
ONE_TO_ONE 路径。

### 三级累计

| | psum (Gbps) | |
|---|---:|---|
| 默认 | 162 | |
| `+ host_reduction=false` | 257 | +58% |
| `+ max_reduction_shard_size=16Mi` | 331 | +29% |
| | | **累计 +104%** |

---

## 机制

### `--xla_tpu_use_megascale_host_reduction=false`

改变的是 lowering，不是调度。

| | 默认 `=true` | `=false` |
|---|---|---|
| megascale transfer type | 单个 `ALL_REDUCE` | `ALL_TO_ALL` + `ALL_GATHER` |
| HLO 中 `add`/`reduce` 指令 | 0 条 | 3 条 |
| 求和执行位置 | host CPU / DRAM | TPU HBM |
| 传输语义 | 融合归约 | 纯 DMA |

默认路径把 SUM 折进 host transfer，因此一个 DCN all-reduce 的 HLO 里没有任何
加法指令——归约不在图里，而在 host 侧的 libtpu 内部。数据被拉进 host DRAM、
由 CPU 线程求和、再推回。host 内存带宽成为瓶颈，此时 DCN 链路仍有一半余量。

关闭后 XLA 发射 reduce-scatter（以 `ALL_TO_ALL` 实现）+ all-gather，两段均为
纯 DMA，加法在 TPU 上执行。

查看 HLO 需用 `after_optimizations_before_buffer_assignment`；
`after_optimizations` 中控制依赖已被 scheduling 吸收。

### `--megascale_max_reduction_shard_size=16777216`

这是运行时参数，HLO 不变。`GraphBuilder` 切分 `ALL_TO_ALL` / `ALL_GATHER` 的
shard 时使用 `max_reduction_shard_size`（默认 8 MiB），而非 `chunk_size`。
因此 `--megascale_chunk_size` 对 all-reduce 路径无效，只对 ONE_TO_ONE 生效。

instrumentation 实测的实际 wire 大小：

```
base                 ppermute:  8 MiB × 2662    psum:  8 MiB × 2667
mrss=16Mi            ppermute:  8 MiB × 2661    psum: 16 MiB × 1334
mrss=16Mi+chunk=16Mi ppermute: 16 MiB × 1330    psum: 16 MiB × 1331
```

`max_reduction_shard_size` 单独即可将 psum 的 wire 传输提到 16 MiB，条数相应减半，
总字节守恒。`ApplyChunking` 不会将其切回 8 MiB。两个 flag 分别控制两条路径，正交。

---

## 已排除的假设

以下均在 tpu7x 上实测排除，附证伪依据。

| 假设 | 证伪依据 |
|---|---|
| DCN 链路本身打不满 | neper `tcp_stream -rw` 天花板 379.2 Gbps；`exchange_add` 可达 334 |
| 双网卡未用满 | 所有变体 eth1/eth2 各 50%，RX/TX = 1.00，TX 与理论值比 1.00 |
| NUMA 跨 socket 开销 | tpu7x 为双 socket，但 6 种 neper 亲和组合全落在 189.5–189.9 Gbps |
| Chaotic Good 未启用 | libtpu 默认自注入；显式关闭反而劣化 4.8× |
| gRPC 参数未调优 | 52+ 组合。唯一有效的 4-flag combo 在 `host_reduction=false` 后从 +10% 塌到 +2% |
| `send_done → recv_done` 控制依赖导致半双工 | 见下节 |
| `ALL_TO_ALL` 与 `ALL_GATHER` 之间的阶段屏障 | 见下节 |
| 传输层存在额外内存拷贝 | instrumentation：`memcpy_needed` 615/616 为 false |
| 发送侧排队延迟 | `queue_latency_us` 均值 0，最大 14 µs，非零占比 2.4% |
| RPC 重试 | `attempt > 0` 占比 0% |

### 控制依赖与阶段屏障

两者均通过定制 libtpu 验证，patch 确认生效，性能无收益。

**控制依赖**：`cross_slice_rewrites.cc` 中 `send_d->AddControlDependencyTo(recv_d)`
使 send 与 recv 严格串行，xprof 证实（op 时长总和 / 时间线并集 = 1.00）。
移除后 `send-done` 从 41.94 ms 塌为 0.00 ms，send 确实变为异步；但时间整体转移到
`recv-done`（63.3 → 169.8 ms），传输总时长 246.4 vs 247.4 ms 基本不变，
`barrier-cores` 从 44 增至 100 ms，总步长反而变长。3 轮实测 +2.0%（p ≈ 0.56）。

结论：TX 与 RX 共享同一底层瓶颈，串行化本身不产生损失。
**neper 的双向天花板 379.2 Gbps 不适用于 MegaScale**——它无法利用双向并发，
真实上限取决于单方向 TPU↔host↔NIC 通路。

**阶段屏障**：`ALL_GATHER` 的发送操作数依赖 TPU 加法输出，加法依赖 `ALL_TO_ALL`
全部接收完成。特化为单阶段后（HLO 确认 `ALL_TO_ALL` 消失），3 轮实测 −2.0%。
跨阶段分块流水（K=4，HLO 确认 send/recv 指令 4 → 16）实测 −2.4%。
xprof 显示阶段间的 TPU 加法仅耗时 0.50 ms。

### `ppermute_uni` 毒化（机制未知）

同一进程内先执行 `ppermute_uni`，后续所有 DCN collective 性能下降：
v6e 5.4×，tpu7x 2.6×，且不恢复。

已排除：dynamic load balancing（`--megascale_grpc_dynamic_lb=false` 无效，
且该 flag 本身无代价）、预映射缓冲区回退（毒化状态下 `memcpy_needed` 仍为
284/284 false）、传输层所有可观测点（不排队、不重试、chunk 不变）。

xprof 定位：仅影响接收侧，`recv-done` ×2.27，`barrier-cores` ×11.5，
`send-done` 基本不变。

**影响**：`dcn/benchmark.py` 默认变体顺序中 `ppermute_uni` 排在首位，
因此使用默认顺序测得的 all-reduce 数值均处于毒化状态。

---

## 测量方法

以下每一条都是踩过的坑。

**变体顺序固定，且不要在 all-reduce 前运行 `ppermute_uni`**。见上节。

**warmups ≥ 200**。σ 从 33% 降至 8%；1000 无额外收益。

**丢弃每个 session 的首次运行**。冷启动系统性偏低 15%。

**多轮必须交替提交**（A,B,A,B），不可 AAA 后 BBB。单轮内曾观测到 6% 的时段漂移。

**同进程内跑一个对照变体**。`exchange_add` 与 `psum` 在 DP=2 下数学等价但走
不同 transport。对照不动才能确认效应真实。注意：验证影响两条路径的 flag 时
（如 `allow_send_recv_overlap`），该对照失效，须比较绝对值。

**报告 best 而非 median**。分布有长右尾。

**判断实际加载的 libtpu 要看 `/proc/*/maps`**，不能看 `importlib.metadata`
——后者报告的是 pip 包版本，与实际映射的 `.so` 无关。

**编译期改动看 HLO，运行时改动看 instrumentation**。
`max_reduction_shard_size` / `chunk_size` 属运行时参数，HLO 完全不变。

**libtpu 的 INFO 日志默认被抑制**。启用 instrumentation 需同时设置
`TPU_STDERR_LOG_LEVEL=0`、`GLOG_stderrthreshold=0` 等，否则采样为 0 行。

**多配置应在单个 pod 内顺序执行**。每个配置单独提交 JobSet 的开销
（Kueue 排队、pip 安装、libtpu 下载）远大于测量本身；单 pod 内 4 配置从提交到
出结果约 10 分钟，分别提交需 1–2 小时。同时天然消除时段漂移。
协调端口需逐次递增，避免残留连接。

---

## 未决

**单方向吞吐的真实上限未知。** neper 双向 379.2 已确认不适用。需要
TPU HBM → host 内存的 d2h DMA 有效带宽、host 内存 → NIC 的单方向上限，
以及两者串联的端到端上限。缺少该基准，无法判断 331 Gbps 距离硬件极限还有多远。

**`ppermute_uni` 毒化机制未知。** 已排除三类假设，见上。

**更大 DP 未测。** 本文全部数据为 DP=2。替代路径为 reduce-scatter + all-gather，
理论上扩展性不劣，但未实测。

**端到端收益未测。** 本文为纯通信 microbenchmark；真实训练步中 all-reduce
与计算存在重叠，端到端提升将小于 104%。

**HBM 压力未评估。** 加法移回 TPU 会占用 HBM 带宽与一份临时 buffer。
dim 32768 下无问题，更大 shard 需确认。

---

## 复现

```bash
scripts/multiconf.sh      # 单 pod 内多配置顺序执行（推荐）
scripts/hostred-stock.sh  # stock libtpu 上验证 host_reduction flag
scripts/chunk-reps.sh     # chunk size 3 轮交替
scripts/xprof-v2.sh       # 抓取 HLO + xprof trace
```

抓 trace 需注意：容器从 upstream 固定 SHA 拉取代码，upstream 无 profiler hook，
脚本以 base64 注入补丁至 `distributed_runner.py`，`benchmark.py` 保持字节不变。

原始日志见 [`results/tpu7x/`](results/tpu7x/)。
