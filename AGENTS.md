# Project: Tool-Call Circuit Discovery

## 1) Project Goal
- 目标：复现并迁移《Knowledge Circuits in Pretrained Transformers》的“定位与寻找”实验思路，用于解释模型为何在下一 token 选择 `<tool_call>`。
- 任务定义：给定成对提示词 `(clean, corrupt)`，关注首个生成 token；`clean` 期望首 token 为 `<tool_call>`，`corrupt` 期望首 token 非 `<tool_call>`。
- 研究范围：只做电路定位/发现与行为解释；不做知识编辑（ROME/FT-M）与应用实验（hallucination/ICL/reverse relation）。

## 2) Critical Data & Folder Semantics
- `/root/data/R3/pair`
  - 主数据集（批量实验入口）。
  - `meta-q*.json`：含 `clean/corrupted`、`segments`、`key_spans`、token 对齐信息。
  - `first_token_len_eval_qwen3_1.7b.csv`：首 token 基线统计。
- `/root/data/R3/sample`
  - R3 内部示例样本（快速调试）。
- `/root/data/R3/sample`
  - 单样本深入分析的默认样本源（强制规则）。
  - 若做 case study / running example，优先使用该目录。
- `/root/data/R3/src`
  - 实验脚本与分析代码。
- `/root/data/R3/figs`
  - 全部图像产出目录。
- `/root/data/R3/reports`
  - 表格、日志、实验结论与中间报告。
- `/root/data/R3/Knowledge Circuits in Pretrained Transformers`
  - 论文 LaTeX 与参考图（对齐实验方法与画图风格的依据）。

## 3) Environment & Runtime Constraints
- 模型：`/root/data/Qwen/Qwen3-1.7B`
- 环境：`base`。
- 计算资源：优先使用 `GPU 4090 24G`。
- 显存策略：若显存不足，不降级实验设计；等待资源释放后再运行。
- 长任务前必须检查 GPU 状态（如 `nvidia-smi`），记录 batch size / dtype / 显存峰值。
- 图像风格最好与参考论文一致
- 只在你的工作目录下读写，不要关注其他代码