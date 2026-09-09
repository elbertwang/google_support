# 融合 all-reduce 慢的原因：求和跑在 host CPU 上

> **本文已被 [`TUNING.md`](TUNING.md) 取代。** 该文是最终结论，包含完整数据、
> 机制、已排除假设清单与测量方法。本文保留为调查过程记录。


```bash
export LIBTPU_INIT_ARGS="--xla_tpu_use_megascale_host_reduction=false"
```

**DCN 上的 `jax.lax.psum` +54%。** stock `libtpu 0.0.44`，不需要自定义二进制，
不需要改图。

测于 tpu7x（2 × `tpu7x-standard-4t`，DP=2，dim 32768，warmups 200，reps 10，best-of）。
`exchange_add` —— 同一个归约写成 `v + ppermute(v)`，走另一条 transport 路径 ——
在每次运行里同进程跟跑，作为进程内对照。

```
                       psum                            exchange_add（对照）
默认           161.1  159.1  166.1  → 162.1 ± 3.6      287.8 ± 10.9
flag=false     245.5  242.9  260.8  → 249.7 ± 9.6      277.5 ± 15.0
                                       +54.0%            −3.6%
```

Welch t ≈ 14.8。对照组不动，所以这不是整机状态漂移。

`psum` 从 **exchange_add 的 56% 提到 90%**；相对同一批网卡上 neper 实测的
379.2 Gbps 双向裸 TCP 天花板，从 43% 提到 66%。

## 原理

这个 flag 改的不是调度，是 **`psum` 的降级方式（lowering）**。

| | 默认 `=true` | `=false` |
|---|---|---|
| megascale transfer type | 单个 `ALL_REDUCE` | `ALL_TO_ALL` + `ALL_GATHER` |
| HLO 里的 `add`/`reduce` 指令 | **0 条** | **3 条** |
| 求和在哪执行 | host CPU / DRAM | TPU HBM |
| 传输语义 | 融合归约 | 纯 DMA |

默认情况下 SUM 被折进 host transfer。这就是为什么一个 DCN all-reduce 的 HLO 里
**一条 add 指令都没有** —— 归约根本不在图里，它在 host 侧的 libtpu 内部。
数据被拉进 host DRAM、由 CPU 线程求和、再推回去。host 内存带宽成了瓶颈，
而此时 DCN 链路还有一半容量闲着。

关掉之后，XLA 吐出 reduce-scatter（以 `ALL_TO_ALL` 实现）+ all-gather。
两段都是纯 DMA，加法在 TPU 上、吃 HBM 带宽。

看 HLO 要看 `after_optimizations_before_buffer_assignment`，不是
`after_optimizations` —— 控制依赖在后者之前已被 scheduling 吸收，看不见。

## 手写 ring 其实一直在做的就是这件事

[`MANUAL-AR.md`](MANUAL-AR.md) 记录了手写 ring 相对 `psum` 的优势：
DP=2 +66.4%、DP=4 +59.7%、DP=8 +37.9%、DP=10 +51.6%。这个优势从来不是 ring 算法带来的。
ring 由 `ppermute` 搭成，而 `ppermute` 是纯 DMA，所以它**一直在绕开 host reduction** ——
和这个 flag 做的事完全一样，只不过 flag 让编译器自己去做。

所以我们听到的那个反对意见 —— *"梯度 all-reduce 是 JAX 自动生成的，自己写一个 ring
不太可能有意义"* —— **是对的，而且现在也不重要了**。你不需要改图，你需要一个 flag。

它还解开了一个一直对不上的结果：`psum_scatter` + `all_gather`
（`manual_ar.py` 里的 `rs_ag`）在 DP=2 实测 **−21%**，尽管它是"正确地"分解了 all-reduce。
原因是 `psum_scatter` 降级成融合的 `REDUCE_SCATTER` transfer，它**同样**走 host reduction。
分解 collective 不是重点，**把求和从 host 拿走**才是。

## 同批测的另外三个 flag

以 patched libtpu 为底座筛查（基线 `psum` 164.7 ± 3.2）：

| flag | psum | 判定 |
|---|---|---|
| `--xla_tpu_use_megascale_host_reduction=false` | 267.6 | **+62.5%，有效** |
| `--megascale_compute_engine_use_executor_for_grpc=false` | 165.3 | +0.4%，无效 |
| 两者叠加 | 244.0 | **比单用 host_reduction 更差** |
| `--megascale_compression_threshold=0` + e5m2 量化 | 崩溃 | peer worker 报不可恢复错误，coordinator abort |

不要上 `compute_engine_use_executor_for_grpc=false`。它的作用是把 Eigen 求和线程
与 gRPC 线程池解耦 —— 但一旦 host reduction 关掉，host 上就没有 Eigen 求和了，
它无事可解耦，只剩线程池碎片化的坏处。两个 flag 直接冲突：叠加后 `psum` 从
267.6 掉到 244.0，对照组同步下滑（285.7 → 248.7），说明它拖的是整个传输层。

FP8 那一档稳定崩溃。报错的是 peer worker，它的 Pod 在日志被读到之前就消失了，
所以根因没抓到。它另有实实在在的数值代价 —— `e5m2` 只有 2 位尾数 ——
所以在梯度 all-reduce 上开它是一个训练精度决策，不是白捡的带宽。

## 它和 gRPC combo 不叠加

[`results/tpu7x/FLOWCTRL.md`](results/tpu7x/FLOWCTRL.md) 里找到了唯一一个
经得起重复的 flag 组合 —— `num_channels=32` +
`dynamic_lb_max_outstanding_bytes=64Mi` + `dynamic_lb_min_outstanding_rpcs=128` +
`grpc_enable_rpc_receive_coalescing` —— 当时给 `psum` **+10.0%**。
叠在 host reduction flag 上重测，2 轮交替：

```
                host_reduction=false      + gRPC combo
psum          243.5  252.1 → 247.8      252.2  253.2 → 252.7    +2.0%
exchange_add  266.7  261.4 → 264.1      266.3  247.6 → 257.0    −2.7%
```

**+10% 塌到 +2%，落进噪声** —— host reduction 单独那一档自己的第二个样本（252.1）
就已经和 combo 的两个样本持平了。combo 当初买到的东西，绝大部分是在补偿
host reduction 造成的损失。**上一个 flag，不要上五个。**

（combo 确实还保留了方差更小这个性质，跟 `FLOWCTRL.md` 当时的观察一致。
但不值得为 2% 多引入四个 flag。）

## 这让哪些既有结论失效

**这个分支里所有 flag 的阴性结论，都是在 host reduction 开着的时候测的**，
也就是在一个真瓶颈在别处的系统上测的。去调一个不是瓶颈的传输层，当然没有信号。
46 个 flag 都是这么筛的。现在传输层第一次真的暴露出来
（247.8 / 379.2 = 裸 TCP 天花板的 65%），这些阴性结论值得重跑 ——
但要注意上面那个 combo 正是反方向的警示：**之前有效的，现在未必有效。**

不受影响的（不经过这条降级路径）：neper 裸 TCP 天花板 379.2 Gbps；
双网卡 50/50 与 RX/TX = 1.00；Chaotic Good 默认已开、关掉劣化 4.8x；
warmups ≥ 200 的方法学。

需要重新审视的：`transport_numa_node` 当初判为负优化，但那个结论针对的是
host 上 Eigen 求和线程的亲和性。现在 host 上已经没有求和了。

## 为什么 46 个 flag 的黑盒筛查漏掉了它

三个原因，第三个才是关键。

1. 我们依据的那份公开 XLA flags 文档里，megascale flag 数量是 **0** ——
   它讲的是 performance / debugging / sparsecore，完全是另一个族。
2. 我们的候选是按 `megascale_*` 前缀猜出来的。而这个 flag 是 `xla_tpu_*`，
   它坐在 XLA → MegaScale 的**降级边界**上，不在 MegaScale runtime 里。
3. **我们测过语义正确的那个 flag，而它是个死开关。**

```
--megascale_use_top_level_all_gather_and_local_reduction_for_ar=true
  164.3 Gbps    vs    baseline 163.1
```

这个名字读作*"用 top-level all-gather + 本地归约来做 AR"* ——
正是最终生效的那个机制。它纹丝不动，于是我们据此把"把归约从 host 拿走"
这整条思路判了死刑。**一个杀死了正确假设的假阴性。**

这就是黑盒 flag 筛查的天花板：它区分不了*"这个 flag 无效"*和*"这个 flag 是死代码"*。
只有读降级路径的源码能区分。找到这个 flag 的功劳属于对
`LowerAllReduceWithShuffle` 的逐行核查。

## 上生产之前还要验的

本文测的范围：只有 DP=2，只有 microbenchmark。

1. **更大的 DP 没测。** 替代路径是 reduce-scatter + all-gather，理论上扩展性
   至少不差，但那只是理论。
2. **端到端收益一定小于 54%。** 这是纯通信的 microbenchmark；真实训练步里
   all-reduce 与 compute 有 overlap。
3. **与 MaxText 默认 flag 的交互**（`xla_tpu_overlap_compute_collective_tc=true`
   那一批）没测。
4. **HBM 压力。** 把加法搬回 TPU 会占 HBM 带宽和一份临时 buffer。
   dim 32768 下没问题，用你真实的 shard 尺寸再确认一遍。

数值正确性已验证：`manual_ar.py` 的 `verify()` 把每个变体与 `psum` 对拍 checksum，
容差 1e-3，开着这个 flag 也通过。

## 复现

```bash
scripts/srcflags.sh        # 四个 flag 的筛查
scripts/hostred-stock.sh   # stock libtpu 上 3 轮重复 + FP8 诊断
scripts/interact.sh        # gRPC combo 还叠不叠加
```

日志在 [`results/tpu7x/hostred/`](results/tpu7x/hostred/)、
[`results/tpu7x/srcflags/`](results/tpu7x/srcflags/)、
[`results/tpu7x/interact/`](results/tpu7x/interact/)。

另外，一个移除 `send_done → recv_done` 控制依赖的 patched libtpu 也做了评估，
它**解释不了**这里的任何现象 —— 见 [`PATCHED-LIBTPU.md`](PATCHED-LIBTPU.md)。
