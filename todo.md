# TODO: 将《Knowledge Circuits》定位实验迁移到 Tool-Call 决策任务
> 模型：`/root/data/Qwen/Qwen3-1.7B`
## Part A. 目标（只做定位/寻找，不做编辑/应用）

### A0. 明确任务与基线（先做）
- 任务：解释模型为何在 `p*`（下一 token 位置）选择或不选择 `<tool_call>`。
- 对照：`clean` 与 `corrupt` 成对输入，且 token 长度已对齐（见 `meta-q*.json/alignment`）。
- 目标 token：`t* = <tool_call>`。
- 首 token 验证基线（来自 `pair/first_token_len_eval_qwen3_1.7b.csv`）：
  - `clean` Top-1 为 `<tool_call>`：147/164（89.63%）
  - `corrupt` Top-1 为 `<tool_call>`：27/164（16.46%）
  - 联合规则（clean 是 `<tool_call>` 且 corrupt 不是）满足：120/164（73.17%）
- 验收：在本项目代码中复算一次上述统计并保存到 `reports/baseline_first_token.csv`。

### A1. 复现论文核心定位流程（ACDC 风格）
- 把模型写成残差重写图：节点包括 Input、Attention heads、MLPs、Output。
- 对边做 zero ablation，计算边重要性（迁移论文 Eq. 1）：
  - `S_clean(e) = log p(t* | clean) - log p(t* | clean, ablate e)`
  - `S_corr(e)  = log p(t* | corr)  - log p(t* | corr,  ablate e)`
  - 对照分数：`DeltaS(e) = S_clean(e) - S_corr(e)`
- 阈值网格参考论文：`tau in {0.02, 0.01, 0.005}`，按拓扑顺序剪枝得到子图 `C_toolcall`。
- 验收：
  - 输出 `reports/circuit_edges.csv`（含 `S_clean/S_corr/DeltaS/keep`）。
  - 输出 `figs/final_circuit.png`（最终发现电路）。

### A2. AP 与 CT 双路径定位（论文思路 + clean/corrupt 适配）
- AP（Activation Patching，节点级）：
  - 在 corrupt 前向中，替换单个头/MLP 在 `p*` 的激活为 clean 对应激活。
  - 记录 `AP(node) = margin_corr_with_patch - margin_corr`。
- CT（Causal Tracing，边/路径级）：
  - 在 corrupt 前向中仅恢复某条源->目标贡献（edge-level patch），追踪因果路径。
  - 记录 `CT(edge) = margin_corr_with_edge_patch - margin_corr`。
- 统一 margin：
  - `margin = logit(t*) - max_{t != t*} logit(t)`。
- 验收图：
  - `figs/ap_head_heatmap.png`（层×头，AP 平均贡献）
  - `figs/ct_head_heatmap.png`（层×头，CT 平均贡献）

### A3. 关键组件行为验证（对应论文“special heads”分析）
- 从 AP/CT Top-K 组件中选关键头（如 `L7H14` 类型）与关键 MLP。
- 对每个关键头做 probe：
  - 注意力模式（attn pattern）
  - 输出写入词表后的 Top-K token/logit
  - 单点 ablation 前后对 `<tool_call>` 概率/排名影响
- 验收图：
  - `figs/L{layer}H{head}_probe.png`（例：`figs/L7H14_probe.png`）

### A4. 复现论文中的“结构结论”到本任务
- 层分布图（参考论文 `layer_distribute`）：
  - 统计被激活组件在各层的频率，分别画 attention 与 MLP 分布。
- 层间 rank/prob 曲线（参考论文 `rank` + `newoutput`）：
  - 追踪 `<tool_call>` 在各层 unembed 后的 rank 与 prob。
  - clean/corrupt 分别绘制；可附对象 token（如 `Here`）竞争轨迹。
- 完整性（completeness）评估（参考论文 Table）：
  - Full model vs Circuit-only vs Random same-size circuit。
  - 指标：`ToolCall@1_clean`、`Reject@1_corr`、`Balanced`。
- 验收：
  - `reports/completeness.csv`
  - `figs/layer_distribution.png`
  - `figs/rank_prob_by_layer.png`

### A5. 单样本 running example（强制样本源）
- 当需要做单样本深挖时，默认使用：`/root/data/R4/sample`。
- 该 case 用于产出：
  - 局部电路可视化
  - 关键头 probe
  - 层间 rank/prob 机制解释

### A6. 明确不做项（本阶段）
- 不做知识编辑（ROME、FT-M）。
- 不做应用向实验（hallucination、ICL、reverse relation、multi-hop editing）。

---

## Part B. 规则（强制执行）

### B1. 命名规范（强制）
- 注意力头：`L{layer}H{head}`，例：`L7H14`
- MLP：`MLP{layer}`，例：`MLP12`
- 其他结点（如 residual）：`RESID_L{layer}`（如确实需要，电路中不出现）
- 文件命名：
  - 热力图：`ap_head_heatmap.png` / `ct_head_heatmap.png`
  - 组件探针图：`L7H14_probe.png`
  - 最终电路图：`final_circuit.png`

### B2. 颜色与对比（强制）
- 所有热力图使用红/蓝发散色系（`RdBu` 或同类）：
  - 0 值必须在中间（白或浅色）
  - 正负贡献必须可直观看出
  - 对比度要足够强
- 若使用裁剪或归一化（如 `vmax` 截断），图注中必须明确写出处理方式。

### B3. 图排版与风格（强制）
- 统一字体与字号层级：标题 > 坐标轴标签 > tick。
- 留白充足，不拥挤。
- 风格参考论文图（简洁、信息密度高但层次清晰）。

### B4. 指标规则（强制）
- 所有贡献分数同时报告 `clean`、`corrupt` 和对照差值（如 `DeltaS`）。
- 主报告至少包含：
  - `ToolCall@1_clean`
  - `Reject@1_corr`
  - `Balanced`
  - `margin` 相关提升（AP/CT 后）
- 结果表必须附样本数与筛选条件（如是否只统计满足 clean 命中的样本）。

### B5. 数据与样本规则（强制）
- 批量实验默认用：`/root/data/R3/pair`
- 单样本实验默认用：`/root/data/R4/sample`
- clean/corrupt 必须成对分析，不可单边报告因果结论。

### B6. 运行环境规则（强制）
- 环境：`base`
- 优先 GPU：`4090 24G`
- 若显存不足：等待资源释放后再运行，不降低实验标准。

---

## 执行顺序（建议）
1. 基线复算与样本过滤（A0）
2. AP/CT 全量扫描（A2）
3. ACDC 剪枝构图（A1）
4. 关键组件 probe（A3）
5. 层分布 + rank/prob + completeness（A4）
6. R4 单样本 case 报告（A5）
