# Varlen attention 性能优化记录

## 目标与验收口径

- 场景来自工作区 `testcase.md`：Qwen3 3B/4B、T5 Encoder/Decoder/Cross，FP16/BF16 共 10 组。
- 对照为同一台 Ascend910_9362、同一卡上的 `torch_npu.npu_fusion_attention`，CANN 9.2.0。
- 保持公开 `flash_attn_varlen_func` 前反向语义；不把调用 CANN 算子作为本项目 kernel 的性能优化。
- 主指标沿用已有 eager NPU Event 时间，分别统计前向、反向、前向+反向。预热 10 次，每轮 50 次，5 轮取中位数。该时间包含提交间隙，不等于纯 kernel 时间。
- 最新验收口径（用户2026-10-25继续优化时明确）：以每组前反向总耗时验收，加速比CANN/Flash不低于0.9×，争取1.0×持平（等价Flash总耗时不超过CANN的1/0.9≈1.111倍）。前向/反向分项用于定位，继续报告全部10组均值与累计耗时比。保持通用优化、禁止TND各batch等长假设；此前平均5%及逐项5%口径保留为历史记录。
- 精度仍采用仓库的 FP32/低精度双参考规则，检查完整 out/dQ/dK/dV。任何精度失败的性能结果只用于诊断。
- 数据、seed=2026、scale、mask、真实累计长度、dtype 和计时范围保持一致。T5 长度为满足表格总 token 数及最大长度的固定合成分布，完整值保存在报告 JSON。

## E00：精度修复后的基线

报告：`../../benchmark_results/varlen_vs_cann_fixed/results.{json,csv,md}`。
原始 10 组精度全部通过，另有 48 组尾块/混合长度回归通过。
已修复 full-attention 调度的 17/16 块越界，以及 dK/dV 单行尾块误入 GEMV；后续修改必须保留这些修复。

| 场景 | dtype | Flash 前向 μs | CANN 前向 μs | Flash 反向 μs | CANN 反向 μs | Flash F+B μs | CANN F+B μs |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3 3B | fp16 | 345.1 | 172.6 | 1588.9 | 744.7 | 1956.3 | 954.3 |
| Qwen3 3B | bf16 | 333.8 | 170.4 | 1609.3 | 751.7 | 1892.4 | 952.6 |
| Qwen3 4B | fp16 | 548.9 | 310.7 | 1826.9 | 952.4 | 2444.9 | 1290.9 |
| Qwen3 4B | bf16 | 465.5 | 306.8 | 1852.0 | 978.4 | 2371.4 | 1319.9 |
| T5 Encoder | fp16 | 3092.8 | 2787.6 | 13689.2 | 11895.9 | 16789.0 | 14689.9 |
| T5 Encoder | bf16 | 2396.9 | 2908.7 | 13673.0 | 11586.4 | 16066.6 | 14519.0 |
| T5 Decoder | fp16 | 2358.8 | 363.5 | 1799.9 | 1304.8 | 4242.8 | 1657.7 |
| T5 Decoder | bf16 | 2140.9 | 374.9 | 1829.8 | 1294.9 | 4011.1 | 1670.4 |
| T5 Cross | fp16 | 2542.6 | 1772.2 | 12121.5 | 9910.2 | 14617.6 | 11660.3 |
| T5 Cross | bf16 | 1612.8 | 1794.6 | 12115.5 | 9905.8 | 13713.8 | 11682.6 |

## 对照源码与待验证的优化点

下列是待测假设，不代表已经获得加速。

| 编号 | 本项目观察 | ops-transformer 对照/设计方向 | 待验证项 |
|---|---|---|---|
| P01 | optimized backward 每次执行设备 `seqlens_q.equal(seqlens_k)`；general backward 将累计长度 D2H，并将 tiling H2D | `flash_attention_score_grad/op_host/arch22` 的 tiling/dispatch，以及 Host 侧 actual_seq 数组接口 | profiler 分离设备同步、拷贝和 kernel 时间；减少重复元数据处理，不缓存可能原地修改的长度值 |
| P02 | causal backward 每次创建 2048×2048 ones/triu mask | CANN 接口测试已预先提供压缩 mask；本项目输入只有 causal 布尔参数 | 可复用不可变设备 mask，注意 device/stream/graph capture 生命周期 |
| P03 | forward 每次生成 AICPU scheduler metadata；短序列 Decoder 前向远慢于 CANN | `flash_attention_score/op_kernel/arch22/flash_attention_var_len_score*.h` 及对应 tiling | 观察 AICPU 调度、任务粒度、批次/头融合；为短 Q 优化调度而不假设所有真实长度都等于 max |
| P04 | optimized backward 分开启动逻辑上的 dQ、dV、dK matmul；L1/L0 数据与 flag 重用有潜力 | `flash_attention_score_grad/op_kernel/arch22/basic_modules/cube_op_fasg_arch22.h` 的 Cube23Process、双缓冲；`cube_modules/` | 对照输入驻留、Cube/Vector 流水、同步依赖，避免盲目删 flag |
| P05 | Cross Q=3，现有 general backward 沿用大块预处理/原子累加/后处理 | `flash_attention_score_grad_s1s2_bn2*.h`、`flash_attention_score_grad_bngs1s2_b.h` 及 tiling | 查看实际 CANN kernel 分支，评估短 Q 的 head/batch 融合、输出直接写回及 workspace 流量 |

参考路径均相对于 `../../ops-transformer/attention/`；实际安装 CANN 的实现可能与仓库版本不同，需以 profiler 的真实调用为准。

## 流水设计对照

前向参考 `flash_attention_score_s1s2_bn2gs1.h::Process`：QK matmul、`ProcessVec1`（scale/mask/softmax）、PV matmul、`ProcessVec2`（输出更新）跨任务交叠，`extraInfo`使用3个槽，matmul/Vector结果采用双缓冲。缓冲槽数量与预发射距离是两种约束，增加深度时必须同步修改Host workspace、AICPU metadata以及device的状态槽，不能只改一个宏。

本项目同样按 QK → softmax → PV → rescale 分工，QK/PV通过GM中间结果交接，使用跨核flag确认生产完成。E23采样显示剩余短中序列的主要差距集中于搬运、Vector指令与任务调度；Cube计算量接近参考。因此优先减少无效访存和调度，不以Cube利用率一个数字代替整个流水分析。

反向参考 `flash_attention_score_grad_s1s2_bn2gs1s2_basic.h::Process` 及 `basic_modules/cube_op_fasg_arch22.h`：Cube1生成QK与dO·V，Vector重算P并形成dS，随后Cube2/Cube3计算dQ、dK、dV。参考代码对连续两个Cube3调用只在其外侧等待。本项目据此保留内部ping-pong槽依赖、删除两个Cube3之间重复的全量排空；Cube2到Cube3仍必须等待后者会整块重载的两个L0B半区，不能把所有同步一并删除。参考中 `Cube23Process` 有D=64/sparseMode=0的特定门槛，此处D=128没有直接照搬。

| 场景 | 采用的设计方向 | 数据依赖与正确性限制 |
|---|---|---|
| Qwen MHA/GQA | GQA多head共用KV、Cube连续流水、逐batch读取实际prefix | 已撤回等长元数据/直接定位特化；GQA必须独立验证NT-LSE和遮蔽尾行，不能仅凭前向out通过 |
| T5 Encoder | 每序列单CV tile、独占梯度写回 | 仅full MHA、非deterministic且实际序列均非空；满足唯一writer才能去atomic和清零 |
| T5 Decoder | 8 heads融合的Vector前向，避免通用Cube/Vector短任务交互 | Q/K≤4、D=128、MHA，无额外功能；保留实际长度和空行语义 |
| T5 Cross | 短Q反向按batch/head独占输出，KV分块重算 | FP32累加；无需全局梯度累加workspace；Vector前向实测更慢，已撤回 |

这些设计以本地ops-transformer源码作为参考，以实际安装CANN的同轮测量作为性能对照，两者版本不应视作完全相同。

## E35检查点代码取舍（历史基线）

| 优化点 | 当前状态 | 泛化与验证边界 |
|---|---|---|
| 只读causal mask复用、纯Host dispatch条件前置 | 保留 | 不依赖batch内长度相等；变长Q/K仍做必要的实际长度检查 |
| bounded immutable backward tiling缓存 | 保留 | key仅包含scalar配置；不缓存用户prefix内容，每次kernel读取实际长度 |
| 连续Cube3及Cube2→Cube3同步收紧 | 保留 | 保护实际L0B consumer与每槽复用，不依赖固定S或batch |
| 单CV tile直接任务定位、独占梯度覆盖写 | 保留 | 由每序列tile数/唯一writer条件决定；空序列/非独占情况继续原路径 |
| 短Q Vector实现 | 保留 | 按UB容量/向量分组选择，读取真实变长prefix；覆盖空行和尾块，范围外继续原kernel |
| GQA多head tile、LSE尾块/遮蔽修复 | 保留 | head分组为分块策略，batch实际S可不同；独立NT-LSE及完整梯度回归 |
| 行级准确Div倒数复用、删除无用LSE广播 | 保留 | 保持FP32除法，显式恢复局部Vector mask；近似Reciprocal已撤回 |
| eager分发及native梯度分配 | 保留 | custom op、FakeTensor、torch.compile与dispatch mode保留原语义 |
| Host/AICPU/device共用流水深度常量 | 保留 | 深度仍为2；UB边界静态检查，防止多端槽数不一致 |
| 窄causal mask转换 | 保留 | 以实际mask宽度为准，不按整块score转换；最终214项回归及10组完整精度通过，独立速度收益未确认 |
| QK下一块K预取 | 已撤回 | E34→E35完整流程对照中Encoder/Cross前向改善，恢复原QK实现 |
| 等长TND元数据缓存、地址推导、固定三块反向调度 | 已撤回 | 按用户要求取消长度分布特化 |
| 仅FP16短Q连续分核 | 已撤回 | 取消dtype特定调度，恢复原通用任务分配 |
| dK/dV换序、后处理双缓冲 | 已撤回 | 没有证明性能收益，减少无效复杂度 |

历史实验日志保留当时的状态与测量，不代表其中所有实验仍在当前代码中。

## 实验日志

- E00：冻结精度修复后的基线，建立逐场景、逐阶段验收口径。
- E01：完成 Qwen3 3B、T5 Decoder、T5 Cross FP16 的前反向 trace；原始文件在 `../../benchmark_results/profile_baseline/`。每阶段采样 3 次，以下是 kernel 时间均值（μs），不能与无 profiler 的 eager 时间直接混用。
  - Qwen3 3B backward：Equal 4.01、AICPU Fill **706.10**、Triu 42.57、自研 FAG 791.58；CANN FAG 747.06。主要可消除开销是重复构造 mask，不是 matmul 慢一倍。
  - Decoder forward：自研 FAInfer 2791.06、AICPU metadata 30.90；CANN FA 352.65。主要差距确实在 kernel，不能只优化 Python 或 metadata。
  - Decoder backward：AICPU Fill 854.38、Triu 43.37、自研 FAG 877.69；CANN FAG 1294.01。
  - Cross forward：自研 FAInfer 2048.41、metadata 30.95；CANN 1790.15。Cross backward：自研 FAGGeneral 12008.98、CANN 9915.18。
  - 新增 `scripts/profile_varlen_vs_cann.py`，复用精度基准的输入准备；diagnostic 输出明确标记 `accuracy=not_checked`。性能实验最终仍须完整双基准精度验证。
- E02（测量中）：optimized/general backward 复用前向已有 `MakeDeviceTriuMask()` 只读缓存，消除每次 Fill/Triu。缓存初次构造和同步沿用原有实现，预热后计时；未引入长度数据缓存。
  - 将 general 分支的纯 host 条件移到设备 Equal 之前；Q/K 总 token 不等时可直接确定不能走等长 optimized 分支，Cross 不再做无意义 Equal 同步。
  - 参考 CANN compressed-mask 输入复用；改变的是准备开销，数学计算与原有 kernel 不变。
- E03（编译中）：前向任务调度对 `total_q == batch * max_q` 的合法输入直接计算 batch 索引，避免各核反复扫描已跳过的批次。由最大长度上界与总和共同证明等长，未仅凭 max 或 tensor 地址假定长度。非均匀长度继续原有调度。
- E02 测量：`../../benchmark_results/perf_e02/timings.json`，10 warmup、50 iterations、5 repeats。Qwen3 3B FP16 backward 917.1 μs（CANN 754.6）、F+B 1197.9 μs（CANN 971.8）；Decoder backward 992.6 μs（CANN 1300.5）。精度验证将在最终全量检查统一执行。
- E03 测量：`../../benchmark_results/perf_e03/timings.json`，20 iterations、3 repeats。Decoder 前向 FP16 2696.2 / BF16 2493.8 μs，未解决瓶颈；各场景与 E02 波动较大，不据此声称加速。
- E04：对照 `flash_attention_score_grad_s1s2_bn2gs1s2_basic.h` 中两个连续 `Cube3Process`，去掉 dV 与 dK 之间的全套 WaitFlag/SetFlag；保留 Cube3 内各 ping-pong buffer 的逐槽依赖和其他阶段边界。48 组回归通过（`/tmp/fa-perf-e04-regression.log`）。结果在 `../../benchmark_results/perf_e04/timings.json`，收益大多在波动内（Qwen4B FP16 backward 1136.1 vs E03 1144.4 μs，Encoder 未改善）。
  - 核实了 `Cube23Process` 只在参考 basic kernel 的 D=64、sparseMode=0 路径启用；当前用例 D=128，不能直接套用该专用融合。
- E05（验证中）：增加 max Q/K ≤4、D=128、MHA 且 head 为 8 的倍数时的 Vector 前向。每任务合并 8 个 head，实际变长长度在设备读取；Q/K/V 一次装入 UB，FP32 点积/稳定 softmax/PV，输出按原 dtype，LSE 保持 NT。支持 full/right-aligned causal 和完全被遮蔽行；GQA/local/softcap/dropout/ALiBi 继续通用路径。与参考源码的按形状选择模板原则一致，避免短序列消耗通用 Cube/Vector 交互流水。尚不能认定精度或性能达标。
- E05 结果：Decoder FP16/BF16 完整精度均通过，报告 `../../benchmark_results/perf_e05_decoder/results.md`。F+B 分别 1149.98 / 1138.20 μs，CANN 1674.65 / 1695.75 μs；FP16 独立 forward 413.48 μs 尚未持平 380.91 μs。
- E06：短 Vector 路径跳过不需要的 AICPU scheduler；optimized backward 对由总和/最大值可证明等长的输入，以及相同 Tensor 对象的前缀，跳过 Equal；真正非均匀且不同对象的前缀仍逐次比较。缓存最多 16 份只依赖 scalar 配置的 immutable backward tiling（包含设备、形状、scale/softcap、ALiBi batch stride、核数），首次拷贝完成后发布，每次使用记录 stream，淘汰时由 allocator 保护在途使用。
  - `tests/test_flash_attn_npu_v2_short_varlen.py` 新增 16 组：2 dtype ×2 head数 ×2 mask ×2 长度分布，验证 out/dQ/dK/dV 和独立 FP32 NT-LSE，含完全遮蔽行；连同原 48 组全部通过（`/tmp/fa-perf-e06-regression.log`）。性能报告 `../../benchmark_results/perf_e06/timings.json`。
- E07（编译中）：general backward 在 full attention、非 deterministic、每个序列可装进一个 CV tile 时，直接由 task id 定位 batch/head/group，避免每个核扫描其他核任务。仍设备读取实际长度；空序列保留空流水槽，Cube/Vector 同步推进，不缓存长度。参考 host 分块/核间任务分配原则；收益与精度待测。
- E07 结果：16 组短序列回归通过；Cross backward FP16 12103.7 / BF16 12119.1 μs，仍远慢于 CANN，减少扫描并非主要收益来源。原始 `../../benchmark_results/perf_e07/timings.json`。
- E08：等长 packed 输入（Q/K 总数分别等于 batch×max）的 dropout-free、非 paged 前向可复用纯 scalar 配置生成的 scheduler metadata。LRU 最多 32 项，key 含设备和 stream、dtype、形状、scale/softcap、causal/window、ALiBi stride；构造的是独立常量长度，不引用用户前缀；异步生命周期沿用 allocator stream 记录。编译/FakeTensor 路径继续已有 metadata op。
  - Qwen3 3B forward FP16 226.1 / BF16 211.4 μs；Qwen4B 345.5 / 443.1 μs，仍未全部达标。`../../benchmark_results/perf_e08/timings.json`。
- E09（测量中）：大 batch≥128、full、max Q/K≤512 的反向尝试 single-CV-tile general kernel，比较 Encoder 与原 128×128 分段 optimized kernel。不能仅因模板名带 optimized 就认定更快；按实测决定保留。
- E10（编译中）：短 Q≤4、KV≤512、D=128 MHA 的前反向 Vector 专用路径，按 batch/head 独占输出，32 行 KV 分块。前向在线稳定 softmax；反向 Q/dOut/δ 驻留 UB、逐块重算 P/dS，dQ 在 UB 累加，dK/dV 直接写最终 dtype，取消全局梯度 workspace、原子加、清零与后处理。保留 actual lengths、右对齐 causal 和空序列；dropout/local/softcap/ALiBi/GQA 走原路径。此阶段仅表示实现，精度/性能尚待检验。
- E09 结果：Encoder general backward 约 13.48–13.56 ms，仅改模板收益有限；`../../benchmark_results/perf_e09/timings.json`。
- E10 结果与取舍：40 组短 Q / KV 边界 / 空序列回归通过。Cross Vector forward 约 5.75 ms，明显退化，撤销该前向 dispatch，恢复 Cube 路径；Vector backward 约 9.2 ms，低于 CANN 约 9.9 ms，保留并继续完整数据验证。`../../benchmark_results/perf_e10/timings.json`。
- E11：TND full MHA、实际各序列均非空、无 used-length、每序列只有一个 CV tile、非 deterministic 时，Host tiling 标记 `exclusiveGradOutput`。每个梯度元素有唯一 writer，Cube 以覆盖写替代 atomic add，Vector 跳过全量梯度 workspace 清零；输出缩放/转换仍走已有 post，其他情况维持原逻辑。
  - Encoder FP16/BF16 完整精度均通过，`../../benchmark_results/perf_e11_encoder/results.md`。
  - FP16 backward 12043.43 vs CANN 11741.22 μs（慢 2.6%），F+B 14769.61 vs14519.31（慢1.7%）。BF16 backward12179.93 vs11920.69（慢2.2%），F+B14785.14 vs14739.60（慢0.3%）。前向均快于 CANN，当前 Encoder 三阶段满足 5% 持平口径。
- E12（验证中）：GQA group=4 且实际 Q>128，尝试将 Q tile 从128 token×1 head调整为32 token×4 heads。复用同一 KV head，一次 GEMM 的 M仍≤128，减少 causal 无效计算与重复 KV 搬运。Host、AICPU、kernel 的 head/token 分块必须同时修改；AICPU 的重复 head 分块函数改为调用共享 host helper，防止调度数量与执行不一致。
- E12 结果：`../../benchmark_results/perf_e12/results.md`。Cross FP16/BF16 完整精度均通过，F/B/F+B 分别为 1607.89/9127.00/10832.18 和1755.99/9193.21/11014.61 μs，三阶段均低于同轮 CANN。Qwen3B 精度通过；Qwen4B 前向输出通过、三项梯度失败，故该 GQA 分块此时不可交付。
- E13（待测）：前向预发射深度由2改为1；反向将 single-CV general 的选择扩展至有足够 batch×KV-head 任务的 causal 场景，single-writer 标记同步允许 causal。分别检查是否减少浅流水启动/排空成本、是否改善 Qwen 反向；无结果前不认定有效。
- E14：单独检查 GQA 的 NT-LSE，发现 Q=256/320 正确，而129/257/279出现跨 head 覆盖。根因是 `rescale_o.hpp::SubCoreCompute` 从表示完整序列的 `layoutOutput.shape(0)` 取得尾块拷贝长度：279 token 的末块实际23行，却按279行搬运 LSE。改用实际 `qSThisSubBlock`。新增独立 FP32 LSE 加完整 out/dQ/dK/dV 回归，覆盖两种 dtype、full/causal、整块/尾块、变长及 Q/K 不等长；修复结果待验证。
- E14 结果：129/257/279 的独立 LSE probe 最大误差约9.54e-7，等长/整块/尾块精度回归14组通过；不等长 causal 暴露全遮蔽 grouped-head 初始化的 LSE 布局问题。E13 的浅流水和 general causal 反向均未获得性能收益，撤回（Qwen4B 反向约1.57ms，对照0.95ms）。`../../benchmark_results/perf_e14/timings.json`。
- E15：尝试 MHA 64 token、GQA 64 token×2 heads；同步把遮蔽行处理从硬编码128的块索引改为实际 token 起点。性能退化：Qwen3B forward约281–287μs，Encoder约3.39ms；GQA 混合128/129长度前向也失败。因此撤销全部新增 Q/head 分块，恢复原分块，不保留失败方案。原始 `../../benchmark_results/perf_e15/timings.json` 和 `/tmp/fa-perf-e15-regression.log`。
- E16（待最终精度验证）：Cube2→Cube3 不再排空全部 L1/L0A flag，仅等待即将整块重载的两个 L0B 半区；L1/L0A 仍按原逐槽依赖保护。对照 arch22 连续 Cube23 的数据驻留/同步原则，不能跳过这两个 L0B consumer 依赖。`../../benchmark_results/perf_e16/` 保存 eager 时间与 kernel trace。
- E17（编译中）：恢复原 Q/head 分块后试验预发射深度3、4个 workspace/任务状态槽；GM/GL/DM 的 UB 区间同步调整。每 Vector 最多64行，DM 的槽跨度由256收紧为64，避免增加流水槽后挤占 LSE staging；静态边界检查继续保留。当前只是假设，最终以精度和性能决定保留。
- E17 首轮失败：AICPU metadata 的 `fa_metadata_args.h::PRELAUNCH_NUM` 仍为3，而 device 使用4槽，造成 workspace 区间交叠；该轮结果无效。新增 `fa_pipeline.h` 作为 Host、AICPU、device 的共同深度来源，消除三处独立常量不一致的风险，重新构建验证。
- E18（待测）：optimized causal backward 在 Host 可证明等长、序列落在257–384时，原调度每 head 恰含两个各占3 tile的矩形，每个15-tile segment恰有5矩形。Cube/Vector 从自己的 segment直接解码 batch/head/Q块，不再扫描其他核任务；保留原矩形、workspace、atomic与流水。uniform标记纳入 tiling cache key，变长继续原调度。新增不完整末段、多轮分配和同形状 uniform/ragged 交替回归。
- E18 性能：Qwen4B backward FP16 991.1 vs CANN964.5μs，BF16 1024.0 vs985.9μs，均在5%内；Qwen3B BF16亦在5%内，FP16仍慢约6.9%。原始 `../../benchmark_results/perf_e18/timings.json`。增加前向流水深度3没有改善，故后续恢复深度2；统一深度常量和 UB 边界检查保留。
- E19/E20（待测）：对照 `flash_attention_score_s1s2_bn2gs1.h` 的 `SoftmaxFlashV2` / reduce 配置，检查本实现的 tail-column reduction。非256/512列时，旧代码对每64列切片做两级 BlockReduce，再合并每行结果；改为一次 WholeReduce（max 明确 ONLY_VALUE），减少 Vector 指令与 barrier，实际 tail mask/行步长不变，保留FP32计算。E19使用深度3用于隔离测量；E20恢复深度2。不能以简化指令数替代精度和实测结论。
- E18 精度：136组相关回归全部通过，包含新增28组 uniform调度/缓存测试，日志 `/tmp/fa-perf-e18-regression.log`。
- E20 全量结果：10组 out/dQ/dK/dV 全部通过，`../../benchmark_results/perf_e20/results.md`。测量期间无并行编译。WholeReduce 虽然减少显式指令/屏障，但 Encoder/Cross 前向显著变慢（FP16约3.52/3.06ms），不符合目标，撤回该归约修改。Qwen4B backward约1.00–1.01ms，在CANN的5%内；Decoder各阶段明显快于CANN。
- 测量注意：早期探索曾将编译和诊断测量并行执行，可能影响 eager 的 Host 提交间隙，不能把跨实验的几百分点差异认作稳定收益。最终验收禁止并行编译，仅使用同一已构建库、同轮 CANN 对照和完整精度结果。
- E21（待测）：普通 eager autograd.Function 内直接调用相同的 native wrapper 实现，避免再次经过 Python custom_op 分发和别名检查；公开注册 op、fake实现保留。torch.compile、Tensor子类、functorch/dispatch mode继续注册op路径，必须验证编译兼容性与梯度。E22将同时恢复原尾列归约，复测完整场景。
- E21 结果：Decoder forward约94μs（CANN约370μs），显示短算子的提交开销确实明显；Qwen仍未全部达标。`../../benchmark_results/perf_e21/timings.json`。尚需编译/模式兼容性验证。
- E22（编译中）：恢复原 BlockReduce；补齐 `init_outputs.hpp` 的全遮蔽 grouped-head NT-LSE strided写回后，重新测试32 token×4 head分块（限定128<Q≤512）。在输出归一化中每行只计算一次FP32 reciprocal，再乘各通道，保留原sum计算LSE；普通非SplitKV不再生成未使用的广播LSE。目标是削减Vector归一化和写回成本，所有精度/模式/性能仍须复测。
  - eager varlen backward 将3次 Python `empty_like` 合并到已有 native optional-gradient 分配路径；编译/dispatch mode仍使用显式输出张量和原 mutation schema。未缓存或共享可变 workspace。
- E22 精度门槛未过：`Reciprocal` 硬件近似在 FP16 causal 梯度中出现约0.015的最大误差，超过双参考容差，停止性能测量。E23改成每行用准确 `Div(1, sum)` 求倒数，再复用乘法；仍把除法数量从每输出元素减少到每行，禁止通过放宽误差门槛接受近似。

- E23：准确 `Div(1,sum)` 版本通过25项回归（20组GQA边界、2组TorchDispatchMode、3组torch.compile）；原始 `/tmp/fa-perf-e23-smoke.log`。`../../benchmark_results/perf_e23/timings.json` 仅计时：Encoder/Decoder全部阶段满足5%口径；Qwen4B FP16也满足。Qwen3B FP16前向199.9 vs171.4μs、反向797.7 vs747.1μs，Qwen4B BF16前向356.7 vs310.9μs，Cross FP16前向2369.9 vs1797.4μs仍未达标。尚未用该版本重跑10组完整数据精度，不能把小规模回归等同完整验收。
- E23流水采样：`../../benchmark_results/profile_e23/`，Profiler Level1 / PipeUtilization，每阶段3次。Qwen3B FP16前向 Cube计算32.84μs（CANN33.59），MTE2 172.71μs（CANN97.07），Vector78.19μs（CANN56.61）；Cross FP16前向Cube135.09μs（CANN138.58），MTE2 2303.43μs（CANN980.85）。各pipe时间可重叠，不能相加；这些数据支持优先检查访存与流水，不能单凭利用率断定唯一根因。
- E24（验证中）：对可证明等长Q的非SplitKV前向，将轮转任务分配改为连续区间分核，尝试改善相邻head/token任务的访存局部性；每核区间独立计算排空条件。变长Q和SplitKV维持原调度。另将E13遗留的single-writer资格收紧回full attention，与实际保留的Encoder路径一致。

- E25（待编译）：进一步按完整head均分optimized causal backward的矩形，每核内部仍以最多5个矩形/15 tile形成segment。Qwen3B每核最大任务由30矩形降至26，Qwen4B由55降至52；最后一个segment允许不足5矩形。Cube/Vector使用相同区间边界，workspace尺寸及累加方法不变；测试将覆盖不足一段、跨batch和多轮分配。理论任务数减少不能替代实际收益验证。

- E24精度门槛：58项通过后，短Q含空序列的FP16 causal用例失败（Q=[0,1,4,3,2,4]，K=[33,0,1,32,257,512]，H=16），停止计时。单独复现失败；最后一批部分head的out最大误差约0.0089、LSE误差约0.018。该输入Q非均匀，未进入E24新增调度。
  - 代码检查发现E22删除无用LSE广播时同时删除了其隐藏副作用：SDK `BrcbImpl` 会调用 `ResetMask()`。`InvalidLineLSEProcess` 使用位掩码写入+inf后未恢复mask，后续使用`isSetMask=false`的mask转换/softmax指令受影响。修复应在产生局部mask的函数内显式恢复，而不是保留无用广播；回归确认前不将该假设写作已验证根因。
  - 排查期间E25暂时恢复逐元素Div，以区分归一化算术与mask状态；不得只根据先前小规模GQA通过就宣布全部边界精度通过。

- E25对照：恢复逐元素Div后，相同边界用例仍以相同输出误差失败（`/tmp/fa-perf-e25-isolate.log`），因此不能将此失败归因于行倒数复用。
- E26（编译中）：`InvalidLineLSEProcess` 在masked Duplicate完成后显式恢复完整Vector mask；恢复E23的Div行倒数复用。将复测空序列/完全遮蔽→长KV的连续任务、均匀反向调度、GQA与编译/dispatch语义。E24/E25两项调度的收益均仍待验证。

- E26回归结果：189项全部通过（`/tmp/fa-perf-e26-regression.log`），涵盖40项短Q/空行、76项反向尾块/均匀调度、20项GQA、2项dispatch mode、3项torch.compile、48项metadata。原失败用例在显式恢复mask后通过，结合原Div仍失败的对照，确认E22删除Brcb的隐含mask恢复导致了该精度回归。最终保留显式恢复，不依赖广播副作用。

- E24/E25在E26中的性能取舍：`../../benchmark_results/perf_e26/timings.json`。Qwen3B前向204.9/234.9μs、4B344.7/425.8μs均退化；反向3B807.3/812.9μs、4B1089.8/1097.6μs也没有收益。因此撤回Qwen连续分核和按head均分反向，保留原交错分配及E18的直接定位。理论上减少最忙核矩形数并不保证流水/访存更快。
  - Cross FP16连续分核前向1497.4 vs CANN1779.0μs，有收益；BF16则2335.2 vs1815.5μs退化。将连续分核严格限定为FP16、TND、可证明等长Q、maxQ≤4、maxK≤512；其他dtype/形状恢复交错分配。该取舍还需最终完整精度与性能复测。
- E27（编译中）：参考前向 `SoftmaxFlashV2` 的basic-block无广播reduce模式，仅在masked首个KV块、行数为8的倍数、列数为64的倍数、无softcap/dropout/ALiBi时尝试SDK基本块softmax。复用mask转换已结束且两份P-store已完成的32KiB临时区域；直接输出本任务槽的FP32 max/sum。尾块、后续在线更新和额外功能继续原算法。此为实际kernel内部实现调整，不是调用CANN融合attention替代本接口。

- E27结果：30项GQA/均匀调度/空行定向回归通过；SDK basic-block未改善Qwen3B FP16（216.6 vs CANN170.2μs），不保留这项实现。该轮`--phase forward`改变了跨case分配/释放顺序，未改动的Encoder也有较大波动，因此不能把后续case与此前三阶段运行的差异全部归因于SDK。原始 `../../benchmark_results/perf_e27/timings.json`；最终验收统一执行三阶段完整脚本。
- E28（编译中）：移除E27 SDK分支，尝试紧凑的QK/P中间矩阵行跨度：从固定512列改为当前实际KV块长度向32列对齐；延迟PV从自己的descriptor重建相同行跨度。每槽最大workspace容量保持不变，Host/AICPU槽起点不变，避免以缩小容量引入跨任务覆盖。对照参考实现的`s2AlignedSize`布局，目标是改善小于512列时的GM访问与缓存局部性。

- E29（待编译）：参考arch22 basic反向的dQ→dK→dV顺序，将连续Cube3改为先dK后dV。dQ、dK均读取dS，尝试缩短其重复读取距离；数学操作、独立输出、全部buffer consumer依赖不变。此时运行中的E28二进制尚未包含该改动，后续将单独对比并决定保留。

- E28结果：30项定向精度通过，但完整顺序下Qwen3B前向230.8/285.6μs、4B343.9/377.2μs均未改善，Cross FP16也退化至2278.0μs；撤回紧凑行跨度，恢复512列固定pitch。原始 `../../benchmark_results/perf_e28/timings.json`。较小的逻辑存储范围并不必然产生更快的硬件访存。
- E30（编译中，包含独立的E29反向顺序实验）：继续沿Scalar/GM读取方向，前向kernel每次启动仅由prefix最后值和batch×max证明Q/K等长；等长任务用乘法恢复prefix偏移，不再逐head/token tile反复从GM读取同样的长度。变长仍逐batch读取，合并重复KV prefix读取；paged/append KV不采用KV等长推断。恢复E28之前的固定中间pitch。
- 待验证P06：causal mask通常只覆盖当前score块右侧一部分；现有两次mask Cast按完整score列数执行，而CopyMask/ApplyMask采用较窄的maskColumnRound。应按实际mask宽度转换，避免对未使用区域做byte→half→float转换，尤其GQA的32-token块；须检查尾列及局部窗口路径后单独验证。

- E30回归：189项全部通过（242.15秒），日志归档 `../../benchmark_results/perf_e30/validation/regression.log`。完整10组双参考精度与三阶段性能仍在运行；此结果不代表已经持平。

- E31（源码已准备，尚未构建）：causal角区mask转换宽度改为CopyMask实际使用的maskColumnRound，保留全部UB别名同步；SWA路径不变。与E32一起构建，但分别用前向/反向独立阶段判断收益。
- E32（源码已准备，尚未构建）：参考`flash_attention_score_grad_post.h::Process`的TQue生产/消费依赖，将optimized backward后处理改为两个FP32输入槽和两个半精度输出槽，预取下一块；总UB占用、每块元素数、核间分工、缩放与CAST_ROUND不变。旧实现input分配2×ubBase但只使用第一半，output分配ubBase但只需其一半，故双缓冲无需增加UB容量。移除逐块MTE3→MTE2全排空，仍由队列保护Vector读取及MTE3写回的槽生命周期。完整精度及是否真正加速均待验证。

- 待验证P08：前向QK已有两个L1 K槽，但当前循环先搬运当前K块、等待、计算并写回，再开始下一块搬运；对非paged且无copyback的D=128路径，可提前一块搬运K，让MTE2与当前块MTE1/Cube交叠。必须由共享ping-pong状态选槽，保留每槽MTE1→MTE2复用依赖；尚未实现。
- 待验证P09：PV在搬入V后立即等待MTE2，尚未发起P搬入。可把V-ready等待推迟到第一次L1→L0B消费之前，与P搬入和标量地址准备交叠；不能删掉V consumer依赖，也不能复用仍在消费的V槽。尚未实现。

- E30完整结果：10组out/dQ/dK/dV均通过，`../../benchmark_results/perf_e30/results.md`。Encoder/Decoder六个dtype场景阶段组合均在5%以内；Qwen3B前向FP16/BF16为256.2/191.4μs，对照169.8/170.1；4B为411.9/374.8μs，对照307.4/306.4。Cross前向2088.7/3100.5μs，对照1785.7/1811.0，也未持平。反向3B795.2/792.1、4B1009.0/1007.0μs，只有4B FP16以1.0519倍略超口径。E29顺序调整未显示明确反向收益；E30前向较此前诊断仍有显著差距，不能宣称等长prefix优化有效。完整精度脚本与仅计时脚本的内存分配顺序不同，后续需要检验地址/工作空间敏感性，而不是从两轮不一致测量归因于dtype本身。
- E31/E32现已进入构建。性能基线E30已退出后才开始编译，避免Host争用及运行进程加载库被覆盖。

- 诊断流程修正（E32起）：发现旧profile脚本在F/B/F+B三阶段始终保留retained graph，且main作用域的closure跨case延长旧输入生命期；正式benchmark的run_case则按阶段释放图、按case释放输入。修正为独立case/phase作用域，F前先运行并释放一次前反向（对应精度检查分配），B之后释放保留图再测F+B，并对齐jit_compile=False。仍标记accuracy=not_checked。这确认了分配生命周期不同这一测量条件差异，尚不能证明所有前向波动的硬件根因；旧diagnostic与完整benchmark的绝对值不可直接作因果比较。
- E32新增12项不同head维度回归：D=64/192/256×dtype×full/causal，变长GQA输入使每核后处理跨多块及尾块，验证双缓冲复用与dQ→dK→dV切换；加上原189项共201项。

- 用户新增参考仓库：`../../asc-devkit`，当前commit `9dfffb2ba02d3f1fb80e1f4e6b7fc4e4d1a92158`。后续按Ascend910对应dav_c220/2201实现核对搬运、队列、cache hint及同步指令；不将950/SIMT专属特性直接套用到910。已读取仓库说明及GlobalTensor::SetL2CacheHint实现；仓库源码与本机CANN9.2安装头文件需核对兼容性。
- E30平均指标（10组等权几何平均，定义speedup=CANN/Flash）：前向1.06285×，反向1.07355×，F+B 1.07835×。F+B各组加速比算术平均为1.11178×，但10组耗时求和的比值仅1.00823×；Decoder的优势不能掩盖其余场景退化。全部30个阶段项中19项达到5%口径，11项仍超出。

- 待验证P10（AscendC资料驱动）：`asc-devkit/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/l2_cache_mode_config.md`建议让只读取一次的大输入绕过L2，保留需要复用的中间数据。Cross短Q MHA的K/V基本是单次扫描、合计约1GiB；将先采集L2Cache/Memory计数器，对照CANN，验证是否存在明显的cache替换/带宽差距，再试验仅对这类大输入关闭K/V cache hint。Qwen的重复KV读取继续保留默认cache。暂未修改cache设置，不能把此假设当作已证实根因。
- AscendC对照补充：`.../pipeline_scheduling/enable_double_buffer.md`说明用独立缓冲保护搬运/计算交叠；`.../memory_access/avoid_same_address_access.md`说明多个核访问同一512B地址范围可能串行化；`.../avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_2201.md`给出2201的UB bank冲突规则。UB bank冲突、GM同地址冲突与L2 miss是不同问题，后续按对应硬件计数器区分，不混称为访存冲突。

- E31/E32结果：201项回归全部通过（260.67秒），日志与当轮源码/哈希归档于`../../benchmark_results/perf_e32/validation/`。新的生命周期对齐诊断见`../../benchmark_results/perf_e32/timings.json`。Qwen反向3B799.3/800.8、4B1023.6/1026.9μs，未见双缓冲后处理收益，决定撤回E32后处理实现，保留新增12项回归。前向3B251.8/232.4、4B407.8/382.9μs也未持平；E31仅收紧两次Cast的数据范围，精度通过但尚不能宣称稳定性能收益。Cross前向2516.1/2630.6μs，与CANN1774.4/1814.9仍有较大差距，正在采集L2Cache计数器。

- E33（源码已准备，待profile退出后构建）：实现P08，非paged、无需append copyback、D=128的QK提前搬入下一块K。复用原有两份L1 K槽，预取槽由共享ping-pong状态确定；每个槽仍等待MTE1释放，计算仍等待该槽MTE2完成，末块不多预取。page gather和copyback沿用原路径。同步撤回E32后处理，避免保留无实测收益的复杂改动；E31窄mask转换暂保留。此时运行中的L2 profiler仍是E32二进制。

- E34（按用户最新要求，源码已清理）：删除E03/E30的等长TND直接定位/偏移推导、E08等长metadata缓存、E18固定三块causal直接解码及uniform tiling标记，删除基于等长证明跳过Equal的条件，撤销E24短Q仅FP16连续分核。所有TND实际长度继续读取真实prefix；同一个Tensor的identity判定及仅依赖scalar的immutable反向tiling缓存仍对任意变长输入成立。保留不依赖等长的重复prefix读取合并、通用Cube同步和形状分块实现。
- E29未证实收益的dK/dV顺序变化同时撤回；E32后处理也已撤回。新增不同head维度测试经路由检查实际覆盖general backward（optimized dispatch仅D=128），故修正测试名/注释及先前“其他native维度测试post”的解释，保留为兼容性回归。
- E33 QK预取源码保留，尚未构建/验证；下一次统一以E34名称构建，包含上述泛化清理和P08预取。先前E30平均值不代表清理后版本，必须重新完整测量。

- E32 L2 profiling（`../../benchmark_results/profile_e32_l2/`）：筛除metadata/Sub后，每个attention kernel均采样3次。Qwen3B FP16/BF16自研AIC读命中率98.80%/98.77%，CANN97.59%/97.58%；Cross自研均50.68%，CANN54.13%/54.18%。高Qwen命中率不支持“容量miss是其主要差距”的简单判断；Cross只有约3.5个百分点差距，单凭此值也不能证明关闭cache会提速。
- 补充Memory采样在`../../benchmark_results/profile_e32_memory/`：此时native仍为E32、Python已移除E08缓存（E34源码），故仅用于kernel搬运诊断，不当作E32或E34完整性能报告。生成报告的L2带宽列均为0，不能解读为真实L2流量为零；保留原始计数器，不据此计算端到端HBM带宽。后续优先验证明确存在的QK预取流水机会。

- 统计工具补充：正式benchmark的JSON/Markdown新增F/B/F+B等权几何平均、算术平均、累计耗时比、完整覆盖标志和逐项5%数量。仅纳入精度通过的行，缺失/失败行会使完整覆盖为false，不允许用筛掉失败后的均值验收。新增CPU单测检验2×与0.5×的几何均值1×、算术均值1.25×及累计时间比0.8×的区别，并验证精度失败不会伪造平均达标。
- 诊断流程再核对：正式benchmark在每组后调用torch.npu.empty_cache；旧诊断遗漏此步骤。E34起补齐，进一步对齐缓存分配器状态；因此E32此前称“生命周期对齐”仍未完全涵盖分配器清理。完整benchmark始终是验收依据。

- E34完整验收：10组双参考out/dQ/dK/dV全部通过，报告`../../benchmark_results/perf_e34/results.md`。等权几何平均F/B/F+B为0.97691×/1.01073×/1.00945×；算术平均1.25428×/1.02306×/1.03833×；累计耗时比0.78187×/1.01598×/0.97128×。按几何平均5%口径持平，但前向累计时间明显慢于CANN，不能只报告Decoder抬高的算术平均。
- E35（撤回对照，构建中）：撤回E33的QK下一块K预取，恢复原QK文件；E34的Encoder FP16前向3477.7μs、Cross FP16前向3557.3μs均较差，需相同完整流程对比，不能将等长特化撤回与预取效果混为一谈。同时收紧large-batch full路由：仅MHA、非deterministic、无softcap/ALiBi/dropout可选该优化，其他功能保持原dispatch；新增两种dtype的129-batch非等长full GQA回归。该保护不改变10组目标输入的路径。计划重跑214项兼容性回归和完整10组精度/性能。

## 复现实验

以下命令在本工作区的已配置Python环境执行，工作目录为`flash-attention-npu`。使用NPU 1；性能测量期间不要并行编译、运行其他NPU任务或替换扩展库。

```bash
source /usr/local/Ascend/cann/set_env.sh
FLASH_ATTN_BUILD_VERSION=v2 FLASH_ATTN_BUILD_NPU=910 /tmp/fa-benchmark-venv/bin/python setup.py build_ext --inplace
/tmp/fa-benchmark-venv/bin/python scripts/benchmark_varlen_vs_cann.py \
  --device 1 --dtype both --warmup 10 --iterations 50 --repeats 5 \
  --reference-batch 16 --output-dir ../benchmark_results/varlen_verified
```

硬件计数器诊断（不替代精度验收；可选择PipeUtilization、L2Cache、Memory、MemoryUB、ResourceConflictRatio）：

```bash
/tmp/fa-benchmark-venv/bin/python scripts/profile_varlen_vs_cann.py \
  --device 1 --case 'Qwen3 3B' --case 'T5 Cross' --dtype both \
  --trace --metrics L2Cache --output-dir ../benchmark_results/varlen_l2_profile
```

源码快照、native SHA256、构建和测试日志放在每轮`validation/`下；正式报告的JSON也记录native和Python接口哈希。对比实验时必须核对版本，不能把只计时诊断、带profiler采样、不同生命周期/分配器状态的数值混作同一轮验收。

- E34兼容性回归：212项全部通过（269.38秒），包含原201项以及11项KV cache split/append/GQA尾块、分页SWA和无效前缀回归；日志`../../benchmark_results/perf_e34/validation/regression.log`。benchmark统计工具7项CPU单测通过。随后完整10组精度也通过，E34平均指标见上文；最终交付以E35结果为准。

## E35检查点结果

最终报告：[完整精度与性能](../../benchmark_results/perf_e35/results.md)，[JSON与原始计时样本](../../benchmark_results/perf_e35/results.json)。同一构建版本、NPU 1、无并行编译；10/10组out/dQ/dK/dV双参考检查全部通过，214项兼容性回归通过（261.84秒），统计工具7项CPU单测通过。源码、native SHA256、构建日志及可复现回归命令归档于[validation](../../benchmark_results/perf_e35/validation/)。最终核对实现和二进制哈希均未变化。

加速比均定义为CANN耗时/Flash耗时，大于1表示Flash更快；各场景等权，未排除任何失败或慢项。

| 阶段 | 几何平均 | 算术平均 | 累计耗时比 | 单项在5%内 |
|---|---:|---:|---:|---:|
| 前向 | 1.05955× | 1.31045× | 0.95787× | 4/10 |
| 反向 | 1.01608× | 1.02798× | 1.02119× | 6/10 |
| 前反向 | 1.02350× | 1.05073× | 1.00611× | 6/10 |

满足当时用户允许的平均性能持平目标；后续验收改为逐组F+B≥0.9×，见E39。前反向累计时间仅减少约0.61%；前向累计时间仍比CANN多约4.40%。Decoder的优势对均值贡献明显，不能把平均达标解释为逐场景全部达标：Qwen四组的三个阶段仍慢于CANN，Cross前向仍有差距。完整表保留所有逐项结果。

E35撤回对照：Encoder FP16前向从E34的3477.7μs降至2572.5μs；Cross FP16从3557.3μs降至2574.5μs。两轮均采用完整精度及相同计时流程，但仍存在地址/系统波动，不能据一次对照断言唯一硬件原因。没有证据支持保留QK预取的复杂度，最终恢复原QK文件。新增的MHA路由保护不改变这10组输入的路由，并通过两项129-batch非等长GQA回归。

最终没有新增等长TND假设、固定三块causal解码、均匀长度metadata缓存或FP16短Q专属调度。保留基于实际tile容量、head分组及唯一writer条件的优化；每个batch读取真实prefix。覆盖变长/空序列、Q/K不等长、不同head维度、GQA、分页/append KV、滑窗、dispatch mode和torch.compile的回归未发现功能退化；未测场景的性能不能由这些正确性回归保证。

P09的PV等待位置调整、P10的L2 hint等仍是未实施假设，已记录而未加入最终版本；没有用未经验证的指令改动填充优化项。


## E36起：检查点之后的通用优化

- 已按用户要求将E35提交为Git检查点`4fabafb`。提交前核对实现与已验收SHA256一致；提交包含29文件、精度修复、通用优化、基准/诊断脚本、回归与实验记录。仓库未配置作者，单次提交使用Codex身份，未改变全局Git配置。
- E36（编译中）：前向kernel只在同一次launch内复用当前batch实际读取的Q/K prefix及长度，batch变化即刷新；不推导等长、不跨调用缓存用户数据。任务扫描中的KV读取从未被GetQSBlockTile使用，移除该冗余读取。
- E36：SetVecMask/SetMask将逐位构造掩码改为低位移位公式，保留len=0/64/128语义；2个函数×129个合法长度，258项CPU穷举等价检查通过。
- E36：PV去掉独立V-ready的MTE2→MTE1 Set/Wait。后续第一个P-ready事件覆盖同一MTE2流水中全部前序V搬运，并在第一次V的L1→L0B消费前等待；V槽复用EVENT_ID4、P/K槽与L0依赖保留。AscendC SetFlag_WaitFlag_ISASI文档支持该顺序依赖；WaitFlag仅阻塞目的流水，故不能未测即宣称大幅重叠收益。分页/append/空行仍须回归。

- E36结果：71项短Q/空行/GQA/分页/append/SWA回归全部通过（131.49秒）。10组诊断计时见`../../benchmark_results/perf_e36/timings.json`，仍标记accuracy=not_checked。Qwen四组F+B约1270.9/1186.3/1564.7/1572.5μs，无达标证据；当前分核下很多核每个任务即切换batch，同batch缓存利用不足。撤回当前batch缓存及对应扫描改动，避免保留未证实收益的状态复杂度；P09和两处掩码公式继续随E37验证，不能把组合实验当作各自独立提速证据。
- E37（编译中）：优化反向任务扫描。在本轮非本核segment的head起点，只有完整head的真实tile数能装入15-tile flush边界前才跳过；最后一个head仍走原batch切换路径。NO_MASK按s1×s2计数，causal按三角tile数计数，int64计算；Cube/Vector共用helper，原workspace位置、矩形划分、原子加顺序和round边界保持。每轮owner判断改为与预计算targetSegment比较，替代逐矩形动态取模。
- E37 CPU模型：8,393组配置穷举/随机检查每轮Cube矩形、Vector坐标、overFlag与游标状态完全一致，同时逐步验证targetSegment比较与取模等价。模拟扫描循环减少Qwen3B48.3%、4B55.1%、混合变长25.7%，不是NPU速度收益；实际精度/总耗时待测。

- E37完整结果：214项回归通过（260.81秒），10/10完整精度通过。Qwen3B FP16/BF16 F+B加速比0.884×/0.871×，4B0.883×/0.897×，尚未满足最新0.9×逐组目标；Encoder1.022×/1.011×、Decoder2.052×/2.025×、Cross1.032×/1.021×均通过。F+B几何平均1.10683×、累计耗时比1.03747×。原始报告`../../benchmark_results/perf_e37/results.md`。保留通用反向扫描优化，尤其Decoder完整结果明显改善；不将组合实验解释为P09/掩码公式各自独立收益。
- E38（编译中）：普通eager TND、非分页、无dropout/softcap/ALiBi/local时，复用仅含标量配置的immutable tiling。每次kernel从真实Q prefix计算任务总数，沿用原分核、batch游标与最后排空；不缓存实际长度/指针，不假定等长。cache最多16项、按device和所有标量配置区分，首次H2D完成后发布，每次recordStream保护生命周期。省去该路径每次AICPU metadata及用于准备的Sub；公开metadata、显式metadata、compile/mode/capture与额外功能维持原路径。
- E38兼容性设计：在现有padding3使用明确magic，结构大小不变；原Host/AICPU producer完整清零，该标记仅内部TND读取。actual K长度仍在runMainLoop覆盖maxK上界，用于右对齐causal和空行判断。每核新增O(B) prefix扫描有成本，是否获益必须实测。新增4项prefix原地修改/跨stream测试，同一存储与标量key切换非等长、空序列、Q/K不同及128/129、512/513分块边界，校验out/dQ/dK/dV/LSE。

- E38首次加载失败（尚未执行NPU精度）：torch_npu头文件的inline currentStreamCaptureStatus引用动态库未导出的CheckCompatibleImpl。通过nm核对本机libtorch_npu导出的currentStreamCaptureStatusMayInitCtx，改用该接口；此时已有真实NPU输入，device context已存在。构建成功不代表扩展可加载，该失败日志独立归档为import_failed.log，重新构建后验证。

- E38回归到163项通过时，旧metadata spy断言失败（期望普通TND每次调用一次AICPU，实际0次），不是数值失败。将原metadata用例显式切到保留的registered-op路线，继续保留一次调用及全部数值断言；新增4项测试则禁止调用AICPU metadata，证明实际覆盖新路径。未修改生产实现或精度规则，重跑受影响的4项及全部48项metadata、11项KV/SWA，共63项；此前155项其余回归无需重复。

- E38完整结果：218项回归分批覆盖通过（155项未受metadata断言修改影响，随后4+48+11=63项通过）；10/10双参考精度通过。F+B加速比依次为3B0.89872×/0.90990×、4B0.93426×/0.97072×、Encoder0.95150×/0.96983×、Decoder1.93896×/1.93668×、Cross0.94241×/0.93969×。9/10达到0.9×，3B FP16不能用四舍五入冒充达标。报告`../../benchmark_results/perf_e38/results.md`。
- E38元数据规模诊断：`../../benchmark_results/metadata_planning_e38/`，相同输入/相同native调用、AB/BA交替计时、5预热20次3轮；B16/32/64/128/256/512，真实长度混合129/257/279，另测短Q混合0..4。causal/full/short共18组设备规划均更快，B512优势仅约0.3%–2.4%；两路线out/LSE逐元素相同，但此诊断不替代独立精度。结果不支持将完整benchmark中Encoder/Cross变慢简单归因于O(B)扫描，更不能据此直接加入固定batch阈值；最终暂不增加此门槛。原始脚本与样本保留。
- E39（Python改动）：plain_varlen已经确认全部实际Tensor输入及变换模式，且其后未执行metadata操作，直接复用该判断调用同一个native wrapper，省去外层wrapped函数重复扫描类型/模式。不改变native二进制、数学计算、任务分配或缓存范围；公开op和非plain路径仍走原分发。复测4项动态prefix、2项dispatch、3项compile、3项metadata，并重新运行10组完整验收。

## E39最终验收与当前保留实现

[前反向验收表](../../benchmark_results/perf_e39/acceptance.md)、[完整精度/三阶段报告](../../benchmark_results/perf_e39/results.md)、[原始样本与版本哈希](../../benchmark_results/perf_e39/results.json)。10/10组out/dQ/dK/dV双参考检查通过，逐组CANN/Flash F+B均不低于0.9×；最低0.91298×，不是以平均值替代逐组验收。

| 场景 | 精度 | Flash μs | CANN μs | CANN / Flash | ≥0.9× |
|---|---|---:|---:|---:|---|
| Qwen3 3B | fp16 | 1060.4 | 968.1 | 0.91298× | PASS |
| Qwen3 3B | bf16 | 1046.8 | 960.4 | 0.91744× | PASS |
| Qwen3 4B | fp16 | 1406.3 | 1309.0 | 0.93078× | PASS |
| Qwen3 4B | bf16 | 1363.0 | 1303.7 | 0.95650× | PASS |
| T5 Encoder | fp16 | 14863.7 | 14351.6 | 0.96554× | PASS |
| T5 Encoder | bf16 | 15283.2 | 14997.1 | 0.98128× | PASS |
| T5 Decoder | fp16 | 831.7 | 1667.8 | 2.00532× | PASS |
| T5 Decoder | bf16 | 834.7 | 1673.0 | 2.00424× | PASS |
| T5 Cross | fp16 | 12138.3 | 11682.1 | 0.96242× | PASS |
| T5 Cross | bf16 | 12621.7 | 11708.4 | 0.92764× | PASS |

F+B等权几何平均1.09749×、算术平均1.15641×、累计耗时比0.98651×。因此累计耗时仍比CANN多约1.37%；Decoder优势明显，Qwen尚未全面持平。Encoder/Cross比E37慢的差异如实保留；18组同输入诊断未支持将其唯一归因于prefix扫描，也没有据此添加固定batch阈值。

验证覆盖：E38的218项不同回归已分批通过（155项未变测试及随后63项），E39仅改Python重复分发判断，相关12项再次通过（40.72秒），随后重新完成全部10组独立精度及性能。无新增误差容差、无筛除慢项、无将Flash替换为CANN。最终实现与二进制SHA256核对一致，日志、源码快照与可复现run.sh均在`perf_e39/validation/`。

相对Git检查点4fabafb最终保留：

- E37通用反向head扫描跳过及每轮owner比较；按真实tile数和workspace容量判断，不固定S或batch。
- 两处Vector mask构造闭式替换；PV依赖由后续P-ready覆盖，保留全部槽复用保护。
- E38普通eager TND仅标量immutable tiling复用，实际prefix每次重新读取，取消该路径每次AICPU metadata与Sub准备；其余功能/公开metadata保留原路由。
- E39复用已经完成的eager判断，避免重复分发检查。

已撤回/未加入：E36当前batch prefix缓存未显示Qwen收益，已撤回；相邻任务二元分块只完成CPU调度检查，未在NPU实测，未加入；batch/prefix预算分支没有足够性能依据，未加入。所有当前路径继续支持非等长TND，新增同一prefix张量原地变更和跨stream回归通过。

## E39泛化测量：4096以下与4096边界

本轮仅测量已提交的`7959288`，没有修改实现或重新编译。新增[用例表](../scripts/benchmark_cases/varlen_upto4096.md)及[实际非等长prefix](../scripts/benchmark_cases/varlen_upto4096_lengths.json)。22形状×FP16/BF16共44项，覆盖MHA/GQA/MQA、causal/full、Q/K不等长、D64/128/256；Hq均为8，scale均为1/sqrt(D)，正长度，无dropout/local/softcap/ALiBi。NPU1，同原基准10次预热、50次×5轮中位数，reference_batch=1完整CPU双参考，测量期间无并行编译。

Flash全部44项out/dQ/dK/dV检查通过；其中42项双方均通过精度并取得有效性能对照。Q1025/K513 causal的两种dtype被当前CANN9.2 TND/sparse_mode=3拒绝（每batch Sq>Skv会出现invalid row），保留为不可比，未改mask语义或删掉用例。Flash这两项的检查结果保存在原始日志及`validation/flash_accuracy_from_log.json`。因此本轮对照覆盖不是44/44成功，不应笼统称全部接口测试通过。

| 子集 | 有效对照/应测 | F+B≥0.9× | 几何平均 | 累计耗时比 |
|---|---:|---:|---:|---:|
| 严格小于4096 | 38/40 | 3/38 | 0.72800× | 0.76537× |
| 4096边界 | 4/4 | 4/4 | 0.96860× | 0.96740× |
| 合计 | 42/44 | 7/42 | 0.74807× | 0.80733× |

下表均为独立测量的前反向总耗时加速比CANN/Flash；D128、causal、GQA4的Hq/Hkv=8/2。每组内部S非等长，表中S是实际最大值，B随S变化，不能解释为固定batch的长度扩展曲线。

| 最大S | B | MHA FP16 | MHA BF16 | GQA4 FP16 | GQA4 BF16 |
|---:|---:|---:|---:|---:|---:|
| 64 | 32 | 0.600× | 0.580× | 0.630× | 0.480× |
| 128 | 16 | 0.615× | 0.634× | 0.613× | 0.633× |
| 256 | 16 | 0.613× | 0.622× | 0.618× | 0.610× |
| 512 | 8 | 0.799× | 0.825× | 0.613× | 0.647× |
| 1024 | 4 | 0.848× | 0.824× | 0.758× | 0.868× |
| 2048 | 2 | 0.849× | 0.877× | 0.895× | 0.914× |
| 4096 | 2 | 0.904× | 0.949× | 1.003× | 1.024× |

其他覆盖：full MHA S512为0.826×/0.769×，full GQA S2048为0.767×/0.773×，full MHA S4095为0.783×/0.825×；短Q4/K4095为0.912×/0.919×，GQA Q129/K2049为0.691×/0.705×；D64 MQA为0.760×/0.743×，D256 GQA2为0.788×/0.780×。此处顺序均为FP16/BF16。

结论：原五场景10项的0.9×验收不能外推至整个4096以下范围。本矩阵中的短序列和若干full/GQA场景仍有明显差距；部分差距集中在反向，例如MHA S64 FP16前向90.0/89.1μs（Flash/CANN），反向374.2/194.6μs。GQA S4096 FP16的前向397.7/826.5μs、反向1326.7/920.4μs，前反向合计才达到1.003×；不能把总耗时持平解释为两个阶段都持平。

个别短序列样本波动较大，原始5轮全部保留，统一采用中位数，没有筛选最优样本。严格小于4096的累计耗时比0.76537×对应Flash累计耗时约多30.66%；这是合成用例等次数执行的汇总，不代表生产流量权重。没有旧版本在同一新矩阵的对照，不能将当前与CANN的差距直接认定为此前优化导致的回退。

[泛化汇总及逐项F+B耗时](../../benchmark_results/generalization_upto4096_e39/summary.md) · [完整三阶段报告](../../benchmark_results/generalization_upto4096_e39/results.md) · [F+B CSV](../../benchmark_results/generalization_upto4096_e39/forward_backward.csv) · [原始样本](../../benchmark_results/generalization_upto4096_e39/results.json)。`validation/run.sh`可复现，保留源码版本/二进制哈希、用例副本和完整日志。


## E40起：D128泛化0.95目标与profiling定位

本轮验收继续使用原基准的eager前向+反向连续执行总耗时，CANN/Flash目标≥0.95×、挑战1.0×；原五场景10项另作性能回归。D128泛化矩阵为40应测/38可比，E39只有2项≥0.95（GQA4 S4096两dtype）；严格<4096为36应测/34可比，0项达标。两项CANN拒绝的Q>K causal仍保留不可比，D64/D256用于兼容回归。

E40 profiling使用实际ragged fixtures，新增profile脚本`--cases`/`--lengths-json`参数，数据在`../../benchmark_results/profile_e40_baseline/`。FP16反向3次trace的kernel中位数：

| 用例 | Flash FAG kernel μs | CANN kernel μs | Flash Equal kernel μs |
|---|---:|---:|---:|
| MHA causal S64 | 94.50 | 107.44 | 2.96 |
| GQA4 causal S256 | 130.54 | 148.60 | 2.78 |
| MHA causal S2048 | 432.55 | 346.67 | 3.84 |

`aten::equal`在预热后的trace调用仍约99–121μs，包含`_local_scalar_dense`约60–77μs的主机等待。设备比较只有约3μs，不能用该数值代表完整开销。短序列Flash kernel已快于CANN，但Host bool分发切断提交；长序列还存在kernel差距。general路线另有每次两次prefix D2H和tiling H2D，需要分别消除。

同值独立prefix与alias的诊断仅用于量化同步成本，不修改正式benchmark输入，不作为验收：MHA S64 FP16总耗时368.96→278.65μs，同轮CANN215.11μs；GQA S256为378.76→281.90μs，同轮CANN256.43μs；S2048为651.70→583.55μs，同轮CANN599.74μs。去掉Equal并不足以保证全部短序列达标。保留`alias_diagnostic_original.json`和原始日志；后续仍使用两份独立prefix进行完整精度、性能验证。

候选设计：普通无额外功能的D128用设备读取实际prefix决定opt/general互斥执行；general tiling只缓存shape、声明max长度上界及总token等不可变标量，不缓存prefix值或相等结果，不制造等长序列。空batch、同总token但不同Q/K分段、prefix原地修改与跨stream都纳入回归。该设计随后在E41–E43落地，最终验收见文末。


### 统一cu_seqlens来源和CANN适配计时（用户补充口径）

此前CANN使用计时外准备的CPU tuple，Flash使用NPU int32 prefix，这两种接口准备条件不同，旧数据保留为`host-precomputed`诊断。后续默认`--cann-lengths-mode device-tensor`，两侧从完全相同的cuq/cuk NPU Tensor取值。CANN包装函数每次前向执行`cu.cpu()[1:]`，批量D2H、去掉初始零后将CPU Tensor交给`torch_npu.npu_fusion_attention_v3`；这些适配发生在计时闭包内，未缓存长度值。独立B测量按原生autograd复用前向保存的CPU metadata，没有人为增加额外拷贝。

已实测本机V3：NPU int32/int64长度Tensor均报`tensor must be on CPU`；CPU int32/int64均成功，out/dQ/dK/dV与原int[]接口逐元素相同。因此默认保留与Flash一致的int32源，不增加人为cast；安装文档只列int64，报告明确本机实际验证范围。原始probe脚本及日志在`perf_e41_diagnostic/validation/`。另提供显式`device-to-host`模式（旧API，每call批量D2H转Python列表），不静默切换接口。CPU基准单测11/11通过，验证同一prefix内容更新每次被读取、retained B不添加人工拷贝、旧模式和新模式标签可区分。

### E41设备反向分发与泛化安全检查

普通连续D128、无dropout/local/softcap/ALiBi、非deterministic、非capture路径：同总token并不推断每batch相等，在两个kernel入口分别比较实际B个累计终点，只有匹配的分支进入计算。optimized与general的互补guard位于任何workspace/gradient写入及跨核同步之前。general tiling采用声明max作为上界、精确Tq/Tk作为workspace大小，设备仍读取每batch真实prefix；仅缓存不可变标量blob，初次同步发布和recordStream保护生命周期。原large-batch full-MHA exclusive路径保持原分发。

集成审查修正：optimized接收B+1 prefix，general接收已去首零的B个终点，必须比较相同的B项，不能在general读B+1项；optimized prefix地址改为data_ptr以尊重非零storage offset。补齐prefix一维、同设备、正batch及int32连续性守卫。新回归改变同地址且不变version的prefix，检查equal→unequal→equal、空batch、跨stream和完整梯度，避免任何指针/版本值缓存假设。

首轮新增回归暴露原optimized scheduler空batch缺陷：S=0但q步长仍2，qSeqIdx无法再等于s1BlockNum=0，形成无效矩形并最终越界。修复CubeAddr/VectorAddr在保持当前segment填充位置的同时跳空batch，末尾空batch正确终止。CPU真实header模拟5376配置通过ASan/UBSan；与原scheduler移除空batch后的逐核逐段descriptor完全一致。首轮失败日志及dump保留，没有删除用例。不同Tq/Tk动态prefix的4项及成对分发新增8项NPU回归均已包含在最终226项通过结果中；总token全部为零仍由公开Python接口拒绝，不宣称支持该输入。

E41正常非空用例快速诊断（未检查独立精度，不作最终验收）：FP16总耗时Flash/CANN为S64 MHA270.8/412.8μs、S256 GQA268.8/440.8μs、full S2048 GQA833.9/924.9μs、full S4095 MHA3446.5/3334.2μs。这里CANN已包含同源Tensor的D2H，不能把相对E39旧CANN口径的ratio增长全部称为kernel优化收益。


### E42/E43：full分块选择与临时区复用

E42试验普通D128 full attention超过一个512×512 CV块时使用general scalar路线。optimized长full常形成256×1024 segment，general为512×512；L0同为128×128×128，后者改善Q/K两端复用并减少dK/dV重复atomic更新。以理想S4096为例，每元素dQ/dK/dV累加次数由4/16/16变为8/8/8；这是访存/累加调度差异，不能称Cube FLOPs减少，也不假设真实ragged长度相同。

诊断中MHA full S4095 FP16总耗时由E41的3446.5μs降至2930.1μs，约14.98%；同轮新口径CANN2935.6μs。BF16为2926.1/3160.0μs。GQA full S2048则从E41 FP16 833.9μs变为868.3μs，未显示收益，因此E43只保留多CV块full **MHA** 的路由，GQA继续原optimized路线。该条件使用mask/head关系/CV容量，不按模型、精确S或batch等长分发；原Qwen causal、Encoder单CV和Cross短Q路径保持。

E43复用paired路径的workspace：optimized先提交，general后提交，同stream且predicate互斥，两者分别按各自offset初始化/使用临时区。general只在已有Byte workspace同设备、连续、零storage offset且容量充足时复用，否则正常分配。当前24AIC/48AIV的普通D128路径，opt workspace比最坏512×512 general至少大46,073,344 bytes（43.939MiB），与B、实际ragged长度、head数和两种dtype无关；因此本矩阵不再同时保留两份scratch，峰值为原optimized大小。精确公式与生命周期审查见`perf_e43/validation/shared_workspace_analysis.txt`。未复用输入内容、梯度或可变metadata。

Python同时移除未padding时的全宽out/dQ/dK/dV切片；需要截断时仍保留切片，不改变输出尺寸。独立短序列诊断显示减少约30–40μs的Host提交开销，但长序列有运行波动，不用该诊断代替完整验收。E41c原始17项精度/模式回归通过；E43增加非零storage offset的prefix view检查，完整226项回归与两个矩阵的最终测试均已结束，结果如下。


参考实现与指令依据：本地`ops-transformer/attention/flash_attention_score_grad/op_host/arch22/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp`第72/74行的SameAB 512×512基础CV块及`AdjustCvInner`、对应arch22 kernel的Cube/Vector分段同步；本轮采用已有general kernel实现该分块，未将Flash调用替换为CANN。设备互斥guard遵守`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md`与`intra_core_sync/SetFlag_WaitFlag_ISASI.md`的同步生命周期约束，所有参与核必须在任何collective前一致选路；未删除既有计算阶段的Set/Wait或SyncAll。

E43冻结候选完整NPU回归 **226/226通过**（348.77秒），覆盖原218项及新增8项设备分发回归；新增测试同时采用非零storage offset的连续prefix view。日志`perf_e43/validation/regression.log`，冻结代码（含新增header、测试、fixtures）及SHA256在同目录`sources/`、`source_version.json`。CPU benchmark单测11/11通过，新增CSV口径标签经dry-run六行逐一核对；所有精度规则和容差不变。

## E43最终验收：同源Tensor、计入CANN D2H

2026-10-25，Ascend910_9362 / CANN 9.2.0 / torch_npu 2.7.1.post11，NPU 1。沿用10次预热、每轮50次、5轮中位数；原五场景reference_batch=16，泛化reference_batch=1，均执行完整CPU FP32与低精度双参考。两个矩阵使用同一冻结构建，测量期间无并行编译；[版本与哈希](../../benchmark_results/perf_e43/validation/source_version.json)及[最终一致性检查](../../benchmark_results/perf_e43/validation/frozen_check.json)已归档。

**主指标为独立测量的F+B总耗时比CANN/Flash，≥0.95达标、≥1表示Flash持平或更快。** F+B不是独立F与B计时结果相加。NPU Event记录eager stream耗时，包含主机提交造成的stream空隙；这些结果不是纯kernel耗时。CANN默认V3 Tensor接口，每次前向从与Flash相同的NPU int32 cuq/cuk执行`.cpu()[1:]`，D2H包含在F及F+B计时内；单独B按原生autograd复用保存的CPU metadata。未用预先准备的Host tuple代替本轮Tensor口径。

| 范围 | 双接口有效对照/应测 | ≥0.95 | ≥1 | 最低比值 | 几何平均 | 累计耗时比 |
|---|---:|---:|---:|---:|---:|---:|
| 原五场景×两dtype | 10/10 | 10/10 | 9/10 | 0.95576× | 1.25933× | 1.05182× |
| D128，严格<4096 | 34/36 | 34/34 | 34/34 | 1.00729× | 1.37272× | 1.25842× |
| D128，4096边界 | 4/4 | 4/4 | 4/4 | 1.03357× | 1.09074× | 1.08961× |
| D128合计 | 38/40 | 38/38 | 38/38 | 1.00729× | 1.33990× | 1.21531× |
| D64辅助项 | 2/2 | 2/2 | 2/2 | 1.05564× | 1.06995× | 1.07023× |
| D256辅助项 | 2/2 | 2/2 | 2/2 | 1.02793× | 1.03099× | 1.03101× |

表中D128合计包含前两项，不能重复累加。阈值判断用未舍入数据；累计耗时比为ΣCANN/ΣFlash，各用例等次数执行，不代表生产流量权重。泛化44项中，Flash **44/44** 的out/dQ/dK/dV精度通过，CANN **42/44** 通过并取得性能结果。另两项为Q1025/K513 causal的FP16/BF16：当前CANN对实际Sq>Skv、右下对齐causal存在invalid row的输入报错，保留`runtime_error`，没有改mask、删用例或计入均值。对应Flash精度证据来自完整日志，见[未配对精度记录](../../benchmark_results/generalization_upto4096_e43/validation/flash_accuracy_from_log.json)。泛化脚本因此退出码为1，表示存在这两项CANN执行错误，不能称44项双方全部验收成功。

### 原五场景逐项F+B

| 场景 | dtype | Flash μs | CANN μs | CANN/Flash |
|---|---|---:|---:|---:|
| Qwen3 3B | fp16 | 994.277 | 1132.869 | 1.139× |
| Qwen3 3B | bf16 | 1005.875 | 1155.408 | 1.149× |
| Qwen3 4B | fp16 | 1346.366 | 1489.921 | 1.107× |
| Qwen3 4B | bf16 | 1341.939 | 1508.813 | 1.124× |
| T5 Encoder | fp16 | 14566.418 | 14750.133 | 1.013× |
| T5 Encoder | bf16 | 14940.381 | 15160.868 | 1.015× |
| T5 Decoder | fp16 | 837.788 | 2065.175 | 2.465× |
| T5 Decoder | bf16 | 828.265 | 2069.413 | 2.498× |
| T5 Cross | fp16 | 11917.770 | 12138.329 | 1.019× |
| T5 Cross | bf16 | 12659.503 | 12099.482 | 0.956× |

原五场景均通过0.95目标，Cross BF16尚未持平。仅比较Flash绝对耗时与E39检查点：累计减少1.646%；Decoder FP16增加0.735%（6.112μs）、Cross BF16增加0.299%（37.754μs），其余8项下降。两项小幅增加如实保留，不能声称逐项耗时完全没有上升。

### D128泛化逐项F+B比值

下表causal、Hq=8，GQA4的Hkv=2；每个batch使用fixture中的真实非等长S。B随S变化，不能当作固定batch的长度扩展曲线。

| 最大S | B | MHA FP16 | MHA BF16 | GQA4 FP16 | GQA4 BF16 |
|---:|---:|---:|---:|---:|---:|
| 64 | 32 | 1.218× | 1.511× | 1.183× | 1.181× |
| 128 | 16 | 1.295× | 1.670× | 1.288× | 1.246× |
| 256 | 16 | 1.774× | 1.377× | 1.588× | 1.689× |
| 512 | 8 | 1.605× | 1.468× | 1.725× | 1.843× |
| 1024 | 4 | 1.336× | 1.373× | 1.521× | 1.472× |
| 2048 | 2 | 1.279× | 1.289× | 1.348× | 1.406× |
| 4096 | 2 | 1.062× | 1.034× | 1.133× | 1.139× |

| 其他D128场景 | FP16 | BF16 |
|---|---:|---:|
| MHA full S512 | 1.335× | 1.290× |
| GQA4 full S2048 | 1.039× | 1.014× |
| MHA full S4095 | 1.007× | 1.107× |
| MHA cross Q4 K4095 | 1.181× | 1.150× |
| GQA4 cross Q129 K2049 | 1.839× | 1.679× |
| MHA cross Q1025 K513 | CANN拒绝 | CANN拒绝 |

D128全部38个可比项在本轮均达到总耗时持平目标，最接近边界的是MHA full S4095 FP16，2901.736μs/2922.880μs（Flash/CANN），比值1.0072867。对应38项Flash累计耗时由E39的32719.447μs降至28067.097μs，下降14.219%；这是同输入的跨轮Flash绝对时间诊断，仍存在系统、提交和分配器状态波动。**E39 CANN使用计时外Host tuple，E43计入同源prefix D2H，新旧CANN/Flash比值的提升不能全部归因于Flash优化。**

本轮结论限于测量矩阵和F+B指标。独立反向仍有差距（泛化42个可比项的B几何平均约0.803×），不能把F+B持平解释为前向、反向分别持平，也不保证未测shape的性能。未降低精度容差；226项回归覆盖动态/空prefix、非零storage offset、跨stream、GQA、尾块、D64/D256、metadata、dispatch/compile及KV cache/paged/SWA，均通过。

最终保留设备prefix互斥分发、general标量tiling复用、full MHA多CV分块、成对kernel临时区复用及无padding时消除全宽切片；修复原optimized空batch调度。撤回无收益的full GQA路由。P/dS覆盖QK/dP临时区仅完成分析，未实现或实测，未计入收益；没有引入等长TND假设或固定模型/精确S特例。

[原五场景汇总及E39对照](../../benchmark_results/perf_e43/summary.md) · [原五场景完整精度/三阶段数据](../../benchmark_results/perf_e43/results.md) · [泛化汇总及全部逐项耗时](../../benchmark_results/generalization_upto4096_e43/summary.md) · [泛化完整精度/三阶段数据](../../benchmark_results/generalization_upto4096_e43/results.md) · [泛化F+B CSV](../../benchmark_results/generalization_upto4096_e43/forward_backward.csv) · [泛化原始样本](../../benchmark_results/generalization_upto4096_e43/results.json)。两个目录的`validation/run.sh`保留复现命令，`validation/benchmark.log`保留完整输出；源码快照、构建及回归日志共用`perf_e43/validation/`。

## device_prefix_dispatch撤回对照（2026-10-25）

用户询问能否不加该分支及其性能影响。本轮在独立Python包中编译诊断版host dispatcher，复用E43已编译的device kernel对象；正式源码和扩展库保持E43版本。不能把原profiling中`aten::equal`约99–121μs直接当成撤回的净损失，因为恢复Host比较同时省去了第二个kernel提交、设备prefix扫描和general的Host准备。

区分两个撤回范围：

| 模式 | 设备paired选路 | Host.equal | general标量tiling/缓存 | 长full MHA分块路线 |
|---|---|---|---|---|
| current | 保留 | 普通eligible路径省去 | 保留 | 保留 |
| host_equal | 取消 | 按原条件恢复 | 保留 | 保留 |
| disabled | 取消 | 按原条件恢复 | 取消 | 取消 |

`host_equal`恢复的是需要比较两份prefix时的Host判断；同Tensor、总T不同、已有直接general/short-query路线仍按对应条件选路。不能仅把`paired_prefix_dispatch`置false而继续跳过`equal`，否则同总T、不同Q/K分段会错误进入optimized。`disabled`把整个`device_prefix_dispatch`置false，因此测量的是三项组合撤回，也不是完整回退到E39。

隔离probe确认：equal和同总T但unequal分段两类输入，current无`aten::equal`/`_local_scalar_dense`，另外两模式各出现一次；三模式out/dQ/dK/dV逐元素一致。正式诊断采用原五场景10项及E43可比D128泛化38项，同一输入/真实prefix、同一进程、同一二进制，分别对current与候选进行6轮AB/BA交替（正反顺序各3轮），每块预热10次、测量50次。切模式前同步NPU；B复用同一retained图，F+B每次新建图。CANN V3在每项后单独测F+B，仍用同源NPU int32 prefix，每次`.cpu()[1:]`的D2H包含在计时内。没有拿旧轮CANN耗时作为本次阈值分母。

以下为**前反向总耗时**。几何增幅使用各项逐轮candidate/current比值中位数的几何均值；累计增幅为各项候选时延中位数之和/对应current时延中位数之和−1。正值表示撤回后变慢。

| 范围 | 撤回方式 | 耗时几何增幅 | 累计耗时增幅 | 相对本轮CANN≥0.95 | 最低CANN/Flash |
|---|---|---:|---:|---:|---:|
| 原五场景10项 | 仅撤paired | +1.600% | +0.323% | 10/10 | 0.98271× |
| 原五场景10项 | 整个开关关闭 | +1.615% | +0.333% | 10/10 | 0.98275× |
| D128泛化38项 | 仅撤paired | +11.467% | +6.300% | 38/38 | 0.95927× |
| D128泛化38项 | 整个开关关闭 | +14.612% | +11.701% | 36/38 | 0.85913× |

原五场景累计增幅较小，主要因为耗时占比大的Encoder/Cross没有换路，不能据此判断短序列没有收益。Qwen3B仅撤paired后F+B增加5.43%–6.02%，Qwen4B增加4.05%–4.13%，约增加54–59μs；Decoder反而缩短约1.60%–1.65%。Encoder/Cross两种撤回模式下的配对变化均接近零，作为未换路的波动对照。泛化原paired路线32项仅撤paired后，逐项F+B增加2.29%–22.50%，增加时间的中位数约50.74μs。

| 代表项 | dtype | 撤回方式 | 对应current μs | 撤回后 μs | 配对耗时变化 | CANN/撤回后 |
|---|---|---|---:|---:|---:|---:|
| MHA causal S64 | fp16 | 仅撤paired | 280.684 | 343.896 | +22.50% | 1.22025× |
| MHA causal S128 | bf16 | 仅撤paired | 287.423 | 346.299 | +20.51% | 1.35920× |
| MHA causal S4096 | fp16 | 仅撤paired | 1824.395 | 1876.545 | +2.79% | 1.00568× |
| MHA full S4095 | fp16 | 仅撤paired | 2942.070 | 2942.432 | −0.002% | 1.02290× |
| MHA full S4095 | fp16 | 整个开关关闭 | 2941.136 | 3503.305 | +19.12% | 0.85913× |
| MHA full S4095 | bf16 | 整个开关关闭 | 2925.051 | 3411.419 | +16.65% | 0.90131× |
| GQA4 cross Q129 K2049 | fp16 | 整个开关关闭 | 283.580 | 387.464 | +36.73% | 1.29189× |

各时延列分别取中位数，配对变化取逐轮比值的中位数，两者不一定严格相除一致。`host_equal`最接近0.95门槛的是GQA4 full S2048 FP16（0.95927255×），因此本轮支持“可以只撤设备paired选路、保留其它优化，仍满足已测矩阵0.95目标”，但余量已变小，不能承诺逐项持平或未测shape达标。整个开关关闭则使MHA full S4095两dtype低于0.95，建议将设备选路、标量metadata优化与full MHA分块拆开评估。

48/48项模式一致性诊断通过，检查out/dQ/dK/dV与current的差异（FP16 atol/rtol均0.005，BF16均0.05）；该检查**不是独立CPU双参考精度验收**，也没有改变正式benchmark的精度规则。本轮未修改正式实现，若实际删除路径，仍应对最终源码版本执行相应精度/兼容性回归。原CANN拒绝的Q1025/K513两项及D64/D256辅助项未进入本轮48项矩阵，排除原因与来源哈希明确保存；没有基于本轮速度筛选样本。

[撤回对照汇总及逐项F+B](../../benchmark_results/device_prefix_ab/summary.md) · [完整B/F+B及原始AB/BA样本](../../benchmark_results/device_prefix_ab/measurements/results.json) · [诊断源码差异](../../benchmark_results/device_prefix_ab/validation/diagnostic.patch) · [复现命令](../../benchmark_results/device_prefix_ab/validation/run.sh) · [独立统计复核](../../benchmark_results/device_prefix_ab/validation/profile_review_audit.json)。
