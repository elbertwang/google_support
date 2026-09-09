# 移除 `send_done → recv_done` 依赖的 patched libtpu：无效果

> **本文已被 [`TUNING.md`](TUNING.md) 取代。** 该文是最终结论，包含完整数据、
> 机制、已排除假设清单与测量方法。本文保留为调查过程记录。


被测产物：一个基于内部快照构建的 libtpu，在 MegaScale cross-slice rewrite pass 里
删掉了一行 ——

```cpp
if (!fusion_data.has_value()) {
  CHECK(send_d);
- RETURN_IF_ERROR(send_d->AddControlDependencyTo(recv_d));
+ // 绕过这条控制依赖，应当能解锁 send/recv 同时进行
}
```

当时的理论是：这条依赖把 send 与 recv 串行化了，使双工链路退化为近乎半双工。

**并非如此。** patch 在 HLO 层确认生效，性能变化 +1.6%，落在噪声里。
真正的原因是 host 侧归约 —— 见 [`HOST-REDUCTION.md`](HOST-REDUCTION.md)。

记录在此，是因为阴性结果和阳性结果一样有价值，也因为这套验证方法可以复用。

## patch 确实被加载了

`importlib.metadata` 报的仍然是 pip 包版本（`libtpu 0.0.44`），
它**不能说明实际映射的是哪个 `.so`**。要查进程：

```
$ kubectl exec $POD -- sh -c 'for p in /proc/[0-9]*/maps; do
    grep -ho "/[^ ]*libtpu[^ ]*\.so" $p; done | sort -u'
/tmp/libtpu_patched.so
```

## patch 生效了，而且是选择性的

取自 `after_optimizations_before_buffer_assignment`（控制依赖在
`after_optimizations` 之前已被 scheduling 吸收，在那里看不到）：

| module | transfer type | `recv-done` 的 control-predecessors |
|---|---|---|
| ALL_GATHER  | 融合路径 | `{%send}` |
| ALL_REDUCE ×4 | 融合路径 | `{%send}` |
| ONE_TO_ONE ×2 | `ppermute` | `{%send-done, %send}` |

patch 只落在融合路径上。`ONE_TO_ONE` 保留了这条依赖。

这一点很方便：在同一个二进制、同一个进程里，`psum` 是实验组，
`exchange_add` 就是**同快照、未打 patch 的对照组**。
它提供了我们本来搭不出来的对照 —— 因为我们拿不到同一份源码快照的未打 patch 构建。

## 性能没有变化

3 轮交替，tpu7x，DP=2，dim 32768，warmups 200：

```
psum         (ALL_REDUCE，patch 生效的路径)
   stock    161.1  159.1  166.1   mean=162.1  sd=3.6
   patched  164.8  167.8  161.5   mean=164.7  sd=3.2    +1.6%
exchange_add (ONE_TO_ONE，对照组)
   stock    276.5  288.6  298.2   mean=287.8  sd=10.9
   patched  305.1  281.5  281.9   mean=289.5  sd=13.5    +0.6%
```

Welch t = 0.94，p ≈ 0.4，差值 95% 置信区间 [−3.1%, +6.4%]。
即便真有效果也在 6% 以内，而它本该解释的差距是 **43%**。
对照组只动 +0.6%，也排除了"新快照普遍更快"这种笼统解释。

第一次单跑给出 `psum` 171.2，看着像 +6.8%。那是噪声带的顶端，重复之后不成立。
在 σ ≈ 3.5 的水平下，单次测量分辨不了这个量级。

## 另一条独立的反证

在 stock 的 dump 里，快的和慢的变体带着**同一条**依赖：

```
ONE_TO_ONE   control-predecessors={%send-done, %send}   ppermute_uni,  313-321 Gbps
ONE_TO_ONE   control-predecessors={%send-done, %send}   ppermute_bidi,       237.5
ALL_GATHER   control-predecessors={%send-done, %send}                        221.0
ALL_REDUCE   control-predecessors={%send-done, %send}                        199.4
```

一个在所有变体上都存在的东西，不可能是区分它们的原因。

## 复现

```bash
scripts/patched-ab.sh    # 交替 A/B，每组 3 轮
```
