#!/usr/bin/env python3
"""Tool-call circuit discovery pipeline for Qwen3-1.7B.

Implements TODO parts A0/A1/A2/A3/A4 for paired clean/corrupt prompts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


@dataclass
class PairExample:
    qid: int
    clean_text: str
    corrupt_text: str


@dataclass
class PromptActivationCache:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pos: int
    attn: List[torch.Tensor]
    mlp: List[torch.Tensor]


@dataclass
class PairActivationCache:
    qid: int
    clean: PromptActivationCache
    corrupt: PromptActivationCache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tool-call circuit discovery")
    p.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    p.add_argument("--pair-dir", type=str, default="pair")
    p.add_argument("--report-dir", type=str, default="reports")
    p.add_argument("--fig-dir", type=str, default="figs")
    p.add_argument("--case-dir", type=str, default="/root/data/R4/sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--batch-size-baseline", type=int, default=2)
    p.add_argument("--head-batch-size", type=int, default=8)
    p.add_argument("--analysis-max-pairs", type=int, default=32)
    p.add_argument("--min-analysis-pairs", type=int, default=24)
    p.add_argument("--tau-grid", type=float, nargs="+", default=[0.02, 0.01, 0.005])
    p.add_argument(
        "--non-circuit-retain",
        type=float,
        default=0.7,
        help="Blend coefficient for non-circuit components (1.0 keeps source unchanged, 0.0 full cross-fill).",
    )
    p.add_argument(
        "--max-circuit-edges",
        type=int,
        default=80,
        help="Maximum number of kept edges in the final bidirectional circuit.",
    )
    p.add_argument(
        "--random-trials",
        type=int,
        default=20,
        help="Number of random same-size circuits for completeness averaging.",
    )
    p.add_argument(
        "--completeness-test-max-pairs",
        type=int,
        default=64,
        help=(
            "Maximum balanced holdout pairs for D_test completeness. "
            "0 uses all holdout; if holdout is empty, D_test reuses D_val."
        ),
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def maybe_disable_hf_progress() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def query_gpu() -> Dict[str, str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ]
    try:
        line = (
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            .strip()
            .splitlines()[0]
        )
        name, mem_total, mem_used, util = [x.strip() for x in line.split(",")]
        return {
            "name": name,
            "mem_total": mem_total,
            "mem_used": mem_used,
            "util": util,
        }
    except Exception as e:
        return {"error": str(e)}


def qid_from_meta_path(path: Path) -> int:
    m = re.search(r"meta-q(\d+)\.json$", path.name)
    if not m:
        raise ValueError(f"Bad meta filename: {path}")
    return int(m.group(1))


def load_examples(pair_dir: Path) -> List[PairExample]:
    examples: List[PairExample] = []
    for meta_path in sorted(pair_dir.glob("meta-q*.json"), key=qid_from_meta_path):
        qid = qid_from_meta_path(meta_path)
        meta = json.loads(meta_path.read_text())
        clean_path = pair_dir / meta["clean"]["file"]
        corr_path = pair_dir / meta["corrupted"]["file"]
        examples.append(
            PairExample(
                qid=qid,
                clean_text=clean_path.read_text(),
                corrupt_text=corr_path.read_text(),
            )
        )
    return examples


def dtype_from_str(x: str) -> torch.dtype:
    if x == "float16":
        return torch.float16
    if x == "bfloat16":
        return torch.bfloat16
    raise ValueError(x)


def model_forward(model, **kwargs):
    kwargs.setdefault("use_cache", False)
    return model(**kwargs)


@torch.no_grad()
def last_logits_batch(
    model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_logits: List[torch.Tensor] = []
    all_lengths: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        tok = tokenizer(
            list(chunk),
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        out = model_forward(model, **tok)
        logits = out.logits
        lengths = tok["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(logits.size(0), device=device)
        last = logits[idx, lengths].float()
        all_logits.append(last.cpu())
        all_lengths.append((lengths + 1).cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_lengths, dim=0)


def token_name(tokenizer, token_id: int) -> str:
    tok = tokenizer.convert_ids_to_tokens(int(token_id))
    if tok is None:
        tok = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    return str(tok)


def margin_from_logits(logits: torch.Tensor, tool_id: int) -> torch.Tensor:
    # logits: [B, V]
    logits = logits.float()
    tool = logits[:, tool_id]
    masked = logits.clone()
    masked[:, tool_id] = -1e30
    other = masked.max(dim=-1).values
    return tool - other


def logprob_tool(logits: torch.Tensor, tool_id: int) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1)[:, tool_id]


def build_baseline_table(
    examples: Sequence[PairExample],
    model,
    tokenizer,
    device: torch.device,
    tool_id: int,
    batch_size: int,
) -> pd.DataFrame:
    clean_texts = [x.clean_text for x in examples]
    corr_texts = [x.corrupt_text for x in examples]
    clean_logits, clean_lens = last_logits_batch(model, tokenizer, clean_texts, device, batch_size)
    corr_logits, corr_lens = last_logits_batch(model, tokenizer, corr_texts, device, batch_size)

    clean_top1 = clean_logits.argmax(dim=-1)
    corr_top1 = corr_logits.argmax(dim=-1)
    rows = []
    for i, ex in enumerate(examples):
        rows.append(
            {
                "q": ex.qid,
                "clean_len": int(clean_lens[i].item()),
                "corr_len": int(corr_lens[i].item()),
                "len_diff_corr_minus_clean": int(corr_lens[i].item() - clean_lens[i].item()),
                "clean_top1": token_name(tokenizer, int(clean_top1[i].item())),
                "corr_top1": token_name(tokenizer, int(corr_top1[i].item())),
                "clean_top1_id": int(clean_top1[i].item()),
                "corr_top1_id": int(corr_top1[i].item()),
                "clean_is_tool": int(clean_top1[i].item() == tool_id),
                "corr_is_tool": int(corr_top1[i].item() == tool_id),
                "balanced": int((clean_top1[i].item() == tool_id) and (corr_top1[i].item() != tool_id)),
            }
        )
    return pd.DataFrame(rows).sort_values("q").reset_index(drop=True)


def collect_clean_cache(
    model,
    clean_ids: torch.Tensor,
    clean_mask: torch.Tensor,
    pos: int,
    n_layers: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    attn_clean: List[torch.Tensor] = [torch.empty(0) for _ in range(n_layers)]
    mlp_clean: List[torch.Tensor] = [torch.empty(0) for _ in range(n_layers)]
    hooks = []

    for l in range(n_layers):
        layer = model.model.layers[l]

        def make_attn_hook(layer_idx: int):
            def hook(mod, inp):
                x = inp[0]
                attn_clean[layer_idx] = x[0, pos].detach().float().cpu()

            return hook

        def make_mlp_hook(layer_idx: int):
            def hook(mod, inp, out):
                mlp_clean[layer_idx] = out[0, pos].detach().float().cpu()

            return hook

        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_attn_hook(l)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))

    _ = model_forward(model, input_ids=clean_ids, attention_mask=clean_mask)
    for h in hooks:
        h.remove()

    return attn_clean, mlp_clean


@torch.no_grad()
def run_corr_baseline(
    model,
    corr_ids: torch.Tensor,
    corr_mask: torch.Tensor,
    tool_id: int,
) -> Tuple[float, float, torch.Tensor]:
    logits = model_forward(model, input_ids=corr_ids, attention_mask=corr_mask).logits[:, -1, :].float()
    margin = float(margin_from_logits(logits, tool_id)[0].item())
    logp = float(logprob_tool(logits, tool_id)[0].item())
    return margin, logp, logits


@torch.no_grad()
def run_clean_baseline(
    model,
    clean_ids: torch.Tensor,
    clean_mask: torch.Tensor,
    tool_id: int,
) -> Tuple[float, torch.Tensor]:
    logits = model_forward(model, input_ids=clean_ids, attention_mask=clean_mask).logits[:, -1, :].float()
    logp = float(logprob_tool(logits, tool_id)[0].item())
    return logp, logits


@torch.no_grad()
def head_batch_ap(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    pos: int,
    layer_idx: int,
    clean_vec: torch.Tensor,
    n_heads: int,
    head_dim: int,
    head_batch_size: int,
    tool_id: int,
    baseline_margin: float,
) -> np.ndarray:
    clean_heads = clean_vec.to(ids.device, dtype=model.dtype).view(n_heads, head_dim)
    deltas = np.zeros((n_heads,), dtype=np.float64)

    for st in range(0, n_heads, head_batch_size):
        ed = min(st + head_batch_size, n_heads)
        head_ids = torch.arange(st, ed, device=ids.device)
        bsz = int(ed - st)
        ids_b = ids.repeat(bsz, 1)
        mask_b = mask.repeat(bsz, 1)

        def patch_hook(mod, inp):
            x = inp[0].clone()
            x_pos = x[:, pos, :].view(bsz, n_heads, head_dim)
            idx = torch.arange(bsz, device=x.device)
            x_pos[idx, head_ids, :] = clean_heads[head_ids, :]
            x[:, pos, :] = x_pos.reshape(bsz, n_heads * head_dim)
            return (x,)

        h = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(patch_hook)
        logits = model_forward(model, input_ids=ids_b, attention_mask=mask_b).logits[:, -1, :].float()
        h.remove()

        margins = margin_from_logits(logits, tool_id).cpu().numpy()
        deltas[st:ed] = margins - baseline_margin

    return deltas


@torch.no_grad()
def head_batch_ablate_logprob_delta(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    pos: int,
    layer_idx: int,
    n_heads: int,
    head_dim: int,
    head_batch_size: int,
    tool_id: int,
    baseline_logprob: float,
) -> np.ndarray:
    deltas = np.zeros((n_heads,), dtype=np.float64)
    for st in range(0, n_heads, head_batch_size):
        ed = min(st + head_batch_size, n_heads)
        head_ids = torch.arange(st, ed, device=ids.device)
        bsz = int(ed - st)
        ids_b = ids.repeat(bsz, 1)
        mask_b = mask.repeat(bsz, 1)

        def ablate_hook(mod, inp):
            x = inp[0].clone()
            x_pos = x[:, pos, :].view(bsz, n_heads, head_dim)
            idx = torch.arange(bsz, device=x.device)
            x_pos[idx, head_ids, :] = 0.0
            x[:, pos, :] = x_pos.reshape(bsz, n_heads * head_dim)
            return (x,)

        h = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(ablate_hook)
        logits = model_forward(model, input_ids=ids_b, attention_mask=mask_b).logits[:, -1, :].float()
        h.remove()
        logp = logprob_tool(logits, tool_id).cpu().numpy()
        deltas[st:ed] = baseline_logprob - logp
    return deltas


@torch.no_grad()
def head_batch_ct(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    pos: int,
    layer_idx: int,
    clean_vec: torch.Tensor,
    n_heads: int,
    head_dim: int,
    head_batch_size: int,
    tool_id: int,
    baseline_margin: float,
) -> np.ndarray:
    clean_heads = clean_vec.to(ids.device, dtype=model.dtype).view(n_heads, head_dim)
    deltas = np.zeros((n_heads,), dtype=np.float64)
    for st in range(0, n_heads, head_batch_size):
        ed = min(st + head_batch_size, n_heads)
        head_ids = torch.arange(st, ed, device=ids.device)
        bsz = int(ed - st)
        ids_b = ids.repeat(bsz, 1)
        mask_b = mask.repeat(bsz, 1)

        def ct_hook(mod, inp, out):
            x = inp[0]
            y = out.clone()
            x_pos = x[:, pos, :].view(bsz, n_heads, head_dim)
            idx = torch.arange(bsz, device=x.device)
            corr_sel = x_pos[idx, head_ids, :]
            diff = clean_heads[head_ids, :] - corr_sel

            # mod.weight: [hidden, hidden]. Slice per incoming head chunk.
            w = mod.weight.view(mod.out_features, n_heads, head_dim)
            w_sel = w[:, head_ids, :]  # [hidden, bsz, head_dim]
            delta = torch.einsum("bd,hbd->bh", diff, w_sel)
            y[:, pos, :] = y[:, pos, :] + delta
            return y

        h = model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(ct_hook)
        logits = model_forward(model, input_ids=ids_b, attention_mask=mask_b).logits[:, -1, :].float()
        h.remove()
        margins = margin_from_logits(logits, tool_id).cpu().numpy()
        deltas[st:ed] = margins - baseline_margin

    return deltas


@torch.no_grad()
def mlp_patch_margin_delta(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    pos: int,
    layer_idx: int,
    clean_vec: torch.Tensor,
    tool_id: int,
    baseline_margin: float,
) -> float:
    clean_vec = clean_vec.to(ids.device, dtype=model.dtype)

    def hook(mod, inp, out):
        y = out.clone()
        y[0, pos, :] = clean_vec
        return y

    h = model.model.layers[layer_idx].mlp.register_forward_hook(hook)
    logits = model_forward(model, input_ids=ids, attention_mask=mask).logits[:, -1, :].float()
    h.remove()
    return float(margin_from_logits(logits, tool_id)[0].item() - baseline_margin)


@torch.no_grad()
def mlp_ablate_logprob_delta(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    pos: int,
    layer_idx: int,
    tool_id: int,
    baseline_logprob: float,
) -> float:
    def hook(mod, inp, out):
        y = out.clone()
        y[0, pos, :] = 0.0
        return y

    h = model.model.layers[layer_idx].mlp.register_forward_hook(hook)
    logits = model_forward(model, input_ids=ids, attention_mask=mask).logits[:, -1, :].float()
    h.remove()
    logp = float(logprob_tool(logits, tool_id)[0].item())
    return baseline_logprob - logp


@torch.no_grad()
def evaluate_with_circuit(
    model,
    pair_caches: Sequence[PairActivationCache],
    keep_head: np.ndarray,
    keep_mlp: np.ndarray,
    tool_id: int,
    device: torch.device,
    non_circuit_retain: float,
) -> Dict[str, float]:
    stats = evaluate_with_circuit_detailed(
        model,
        pair_caches,
        keep_head,
        keep_mlp,
        tool_id,
        device,
        non_circuit_retain,
    )
    return {
        "ToolCall@1_clean": float(stats["ToolCall@1_clean"]),
        "Reject@1_corr": float(stats["Reject@1_corr"]),
        "Balanced": float(stats["Balanced"]),
    }


@torch.no_grad()
def evaluate_with_circuit_detailed(
    model,
    pair_caches: Sequence[PairActivationCache],
    keep_head: np.ndarray,
    keep_mlp: np.ndarray,
    tool_id: int,
    device: torch.device,
    non_circuit_retain: float,
) -> Dict[str, object]:
    clean_hits: List[int] = []
    corr_rejects: List[int] = []
    balanced_hits: List[int] = []

    for pair in pair_caches:
        pred_clean = run_mixed_prompt_with_cache(
            model,
            pair.clean,
            pair.corrupt,
            keep_head,
            keep_mlp,
            device,
            non_circuit_retain,
        )
        pred_corr = run_mixed_prompt_with_cache(
            model,
            pair.corrupt,
            pair.clean,
            keep_head,
            keep_mlp,
            device,
            non_circuit_retain,
        )
        is_clean = int(pred_clean == tool_id)
        is_reject = int(pred_corr != tool_id)
        clean_hits.append(is_clean)
        corr_rejects.append(is_reject)
        balanced_hits.append(int(is_clean and is_reject))

    clean_arr = np.array(clean_hits, dtype=np.int32)
    reject_arr = np.array(corr_rejects, dtype=np.int32)
    balanced_arr = np.array(balanced_hits, dtype=np.int32)
    n = max(len(clean_arr), 1)
    return {
        "ToolCall@1_clean": float(clean_arr.mean() if len(clean_arr) > 0 else 0.0),
        "Reject@1_corr": float(reject_arr.mean() if len(reject_arr) > 0 else 0.0),
        "Balanced": float(balanced_arr.mean() if len(balanced_arr) > 0 else 0.0),
        "clean_hits": clean_arr,
        "corr_rejects": reject_arr,
        "balanced_hits": balanced_arr,
        "n": int(n),
    }


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return float(lo), float(hi)


def metrics_from_baseline_slice(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], int]:
    n = int(len(df))
    if n <= 0:
        zero_metrics = {
            "ToolCall@1_clean": 0.0,
            "Reject@1_corr": 0.0,
            "Balanced": 0.0,
        }
        zero_ci = {
            "ToolCall@1_clean_ci": (0.0, 0.0),
            "Reject@1_corr_ci": (0.0, 0.0),
            "Balanced_ci": (0.0, 0.0),
        }
        return zero_metrics, zero_ci, 0

    clean_success = int(df["clean_is_tool"].sum())
    reject_success = int((1 - df["corr_is_tool"]).sum())
    balanced_success = int(df["balanced"].sum())
    metrics = {
        "ToolCall@1_clean": float(clean_success / n),
        "Reject@1_corr": float(reject_success / n),
        "Balanced": float(balanced_success / n),
    }
    ci = {
        "ToolCall@1_clean_ci": wilson_interval(clean_success, n),
        "Reject@1_corr_ci": wilson_interval(reject_success, n),
        "Balanced_ci": wilson_interval(balanced_success, n),
    }
    return metrics, ci, n


def metrics_from_detailed_eval(
    detail: Dict[str, object]
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], int]:
    n = int(detail["n"])
    clean_hits = int(np.asarray(detail["clean_hits"]).sum())
    reject_hits = int(np.asarray(detail["corr_rejects"]).sum())
    balanced_hits = int(np.asarray(detail["balanced_hits"]).sum())
    metrics = {
        "ToolCall@1_clean": float(detail["ToolCall@1_clean"]),
        "Reject@1_corr": float(detail["Reject@1_corr"]),
        "Balanced": float(detail["Balanced"]),
    }
    ci = {
        "ToolCall@1_clean_ci": wilson_interval(clean_hits, n),
        "Reject@1_corr_ci": wilson_interval(reject_hits, n),
        "Balanced_ci": wilson_interval(balanced_hits, n),
    }
    return metrics, ci, n


@torch.no_grad()
def collect_prompt_activation_cache(
    model,
    tokenizer,
    text: str,
    n_layers: int,
    device: torch.device,
) -> PromptActivationCache:
    tok = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = tok["input_ids"].to(device)
    mask = tok["attention_mask"].to(device)
    pos = int(ids.shape[1] - 1)
    attn_cache: List[torch.Tensor] = [torch.empty(0) for _ in range(n_layers)]
    mlp_cache: List[torch.Tensor] = [torch.empty(0) for _ in range(n_layers)]
    hooks = []

    for l in range(n_layers):
        layer = model.model.layers[l]

        def make_attn_hook(layer_idx: int):
            def hook(mod, inp):
                attn_cache[layer_idx] = inp[0][0, pos].detach().float().cpu()

            return hook

        def make_mlp_hook(layer_idx: int):
            def hook(mod, inp, out):
                mlp_cache[layer_idx] = out[0, pos].detach().float().cpu()

            return hook

        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_attn_hook(l)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))

    _ = model_forward(model, input_ids=ids, attention_mask=mask)
    for h in hooks:
        h.remove()

    return PromptActivationCache(
        input_ids=ids.cpu(),
        attention_mask=mask.cpu(),
        pos=pos,
        attn=attn_cache,
        mlp=mlp_cache,
    )


@torch.no_grad()
def build_pair_activation_caches(
    model,
    tokenizer,
    examples: Sequence[PairExample],
    n_layers: int,
    device: torch.device,
) -> List[PairActivationCache]:
    pairs: List[PairActivationCache] = []
    for ex in examples:
        clean = collect_prompt_activation_cache(model, tokenizer, ex.clean_text, n_layers, device)
        corrupt = collect_prompt_activation_cache(model, tokenizer, ex.corrupt_text, n_layers, device)
        if clean.pos != corrupt.pos:
            continue
        pairs.append(PairActivationCache(qid=ex.qid, clean=clean, corrupt=corrupt))
    return pairs


@torch.no_grad()
def run_mixed_prompt_with_cache(
    model,
    prompt: PromptActivationCache,
    fill: PromptActivationCache,
    keep_head: np.ndarray,
    keep_mlp: np.ndarray,
    device: torch.device,
    non_circuit_retain: float,
) -> int:
    n_layers, n_heads = keep_head.shape
    hidden = model.config.hidden_size
    head_dim = hidden // n_heads
    pos = prompt.pos

    ids = prompt.input_ids.to(device)
    mask = prompt.attention_mask.to(device)
    hooks = []
    for l in range(n_layers):
        keep_h = torch.tensor(keep_head[l], device=device, dtype=torch.bool)
        keep_m = bool(keep_mlp[l])
        fill_head = fill.attn[l].to(device=device, dtype=model.dtype).view(n_heads, head_dim)

        def make_head_hook(keep_h_local: torch.Tensor, fill_head_local: torch.Tensor):
            def hook(mod, inp):
                x = inp[0].clone()
                x_pos = x[0, pos, :].view(n_heads, head_dim)
                x_pos[~keep_h_local, :] = (
                    non_circuit_retain * x_pos[~keep_h_local, :]
                    + (1.0 - non_circuit_retain) * fill_head_local[~keep_h_local, :]
                )
                x[0, pos, :] = x_pos.reshape(hidden)
                return (x,)

            return hook

        hooks.append(
            model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                make_head_hook(keep_h, fill_head)
            )
        )

        if not keep_m:
            fill_m = fill.mlp[l].to(device=device, dtype=model.dtype)

            def make_mlp_hook(fill_m_local: torch.Tensor):
                def hook(mod, inp, out):
                    y = out.clone()
                    y[0, pos, :] = (
                        non_circuit_retain * y[0, pos, :]
                        + (1.0 - non_circuit_retain) * fill_m_local
                    )
                    return y

                return hook

            hooks.append(model.model.layers[l].mlp.register_forward_hook(make_mlp_hook(fill_m)))

    out = model_forward(model, input_ids=ids, attention_mask=mask)
    for h in hooks:
        h.remove()
    return int(out.logits[0, -1].float().argmax().item())


def select_bidirectional_indices(
    circuit_df: pd.DataFrame,
    tau: float,
    max_edges: int,
) -> List[int]:
    max_edges = max(2, int(max_edges))
    max_edges = min(max_edges, len(circuit_df))

    pos_df = circuit_df[circuit_df["DeltaS"] >= tau].sort_values("DeltaS", ascending=False)
    neg_df = circuit_df[circuit_df["DeltaS"] <= -tau].sort_values("DeltaS", ascending=True)
    if len(pos_df) == 0:
        pos_df = circuit_df.sort_values("DeltaS", ascending=False).head(1)
    if len(neg_df) == 0:
        neg_df = circuit_df.sort_values("DeltaS", ascending=True).head(1)

    pos_target = max_edges // 2
    neg_target = max_edges - pos_target

    selected = list(pos_df.head(pos_target).index) + list(neg_df.head(neg_target).index)
    selected = list(dict.fromkeys(selected))
    remain = max_edges - len(selected)
    if remain > 0:
        pool = circuit_df.drop(index=selected, errors="ignore").copy()
        pool["abs_delta"] = pool["DeltaS"].abs()
        selected.extend(list(pool.sort_values("abs_delta", ascending=False).head(remain).index))
    return selected


def make_random_circuit(
    n_layers: int,
    n_heads: int,
    n_keep_heads: int,
    n_keep_mlps: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep_head = np.zeros((n_layers, n_heads), dtype=bool)
    keep_mlp = np.zeros((n_layers,), dtype=bool)

    all_head_idx = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    chosen_heads = rng.choice(len(all_head_idx), size=n_keep_heads, replace=False)
    for idx in chosen_heads:
        l, h = all_head_idx[int(idx)]
        keep_head[l, h] = True

    mlp_layers = np.arange(n_layers)
    chosen_mlp = rng.choice(mlp_layers, size=n_keep_mlps, replace=False)
    keep_mlp[chosen_mlp] = True
    return keep_head, keep_mlp


def build_circuit_df(
    ap_head: np.ndarray,
    ct_head: np.ndarray,
    s_clean_head: np.ndarray,
    s_corr_head: np.ndarray,
    ap_mlp: np.ndarray,
    ct_mlp: np.ndarray,
    s_clean_mlp: np.ndarray,
    s_corr_mlp: np.ndarray,
    tau: float,
) -> pd.DataFrame:
    rows = []
    n_layers, n_heads = ap_head.shape
    for l in range(n_layers):
        for h in range(n_heads):
            delta = float(s_clean_head[l, h] - s_corr_head[l, h])
            rows.append(
                {
                    "edge": f"L{l}H{h}->Output",
                    "type": "head",
                    "layer": l,
                    "head": h,
                    "S_clean": float(s_clean_head[l, h]),
                    "S_corr": float(s_corr_head[l, h]),
                    "DeltaS": delta,
                    "AP": float(ap_head[l, h]),
                    "CT": float(ct_head[l, h]),
                    "keep": int(delta >= tau),
                }
            )
        d_mlp = float(s_clean_mlp[l] - s_corr_mlp[l])
        rows.append(
            {
                "edge": f"MLP{l}->Output",
                "type": "mlp",
                "layer": l,
                "head": -1,
                "S_clean": float(s_clean_mlp[l]),
                "S_corr": float(s_corr_mlp[l]),
                "DeltaS": d_mlp,
                "AP": float(ap_mlp[l]),
                "CT": float(ct_mlp[l]),
                "keep": int(d_mlp >= tau),
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(
    mat: np.ndarray,
    title: str,
    out_path: Path,
    center_zero: bool = True,
    cmap: str = "RdBu_r",
) -> Tuple[float, float]:
    abs_q = float(np.percentile(np.abs(mat), 98))
    if abs_q <= 0:
        abs_q = 1e-6
    vmin, vmax = (-abs_q, abs_q) if center_zero else (float(mat.min()), float(mat.max()))

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Score (clipped to +/-{abs_q:.4f})")

    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return vmin, vmax


def plot_layer_distribution(
    keep_head: np.ndarray,
    keep_mlp: np.ndarray,
    out_path: Path,
) -> None:
    n_layers = keep_head.shape[0]
    head_pct = keep_head.sum(axis=1) / max(keep_head.shape[1], 1) * 100.0
    mlp_pct = keep_mlp.astype(float) * 100.0

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(n_layers)
    ax.plot(x, head_pct, marker="o", linewidth=2.2, color="#d62728", label="Attention (% active)")
    ax.plot(x, mlp_pct, marker="s", linewidth=1.9, color="#1f77b4", label="MLP (% active)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Activated Component Ratio (%)")
    ax.set_ylim(-2, 102)
    ax.set_title("Layer-Wise Activated Component Distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


@torch.no_grad()
def layer_rank_prob(
    model,
    tokenizer,
    examples: Sequence[PairExample],
    tool_id: int,
    competitor_id: int | None,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_layers = model.config.num_hidden_layers
    clean_rank = np.zeros(n_layers, dtype=np.float64)
    corr_rank = np.zeros(n_layers, dtype=np.float64)
    clean_prob = np.zeros(n_layers, dtype=np.float64)
    corr_prob = np.zeros(n_layers, dtype=np.float64)
    clean_comp_prob = np.zeros(n_layers, dtype=np.float64)
    corr_comp_prob = np.zeros(n_layers, dtype=np.float64)

    def one_pass(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tok = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        tok = {k: v.to(device) for k, v in tok.items()}
        pos = tok["input_ids"].shape[1] - 1
        out = model_forward(model, **tok, output_hidden_states=True)
        hs = out.hidden_states  # len n_layers+1
        ranks = np.zeros(n_layers, dtype=np.float64)
        probs = np.zeros(n_layers, dtype=np.float64)
        comp_probs = np.zeros(n_layers, dtype=np.float64)
        for l in range(n_layers):
            h = hs[l + 1][:, pos, :]
            h_norm = model.model.norm(h)
            logits = model.lm_head(h_norm).float()[0]
            probs_all = torch.softmax(logits, dim=-1)
            p = probs_all[tool_id].item()
            t = logits[tool_id].item()
            r = int((logits > t).sum().item()) + 1
            ranks[l] = r
            probs[l] = p
            if competitor_id is not None and competitor_id >= 0:
                comp_probs[l] = probs_all[competitor_id].item()
        return ranks, probs, comp_probs

    for ex in examples:
        r1, p1, cp1 = one_pass(ex.clean_text)
        r2, p2, cp2 = one_pass(ex.corrupt_text)
        clean_rank += r1
        clean_prob += p1
        corr_rank += r2
        corr_prob += p2
        clean_comp_prob += cp1
        corr_comp_prob += cp2

    n = max(len(examples), 1)
    return (
        clean_rank / n,
        clean_prob / n,
        corr_rank / n,
        corr_prob / n,
        clean_comp_prob / n,
        corr_comp_prob / n,
    )


def plot_rank_prob(
    clean_rank: np.ndarray,
    clean_prob: np.ndarray,
    corr_rank: np.ndarray,
    corr_prob: np.ndarray,
    clean_comp_prob: np.ndarray,
    corr_comp_prob: np.ndarray,
    competitor_label: str,
    out_path: Path,
) -> None:
    x = np.arange(len(clean_rank))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)

    ax1.plot(x, clean_rank, label="clean", color="#1f77b4", linewidth=2.0)
    ax1.plot(x, corr_rank, label="corrupt", color="#d62728", linewidth=2.0)
    ax1.set_ylabel("Rank (lower better)")
    ax1.set_title("<tool_call> Rank by Layer")
    ax1.legend(frameon=False)

    ax2.plot(x, clean_prob, label="clean", color="#1f77b4", linewidth=2.0)
    ax2.plot(x, corr_prob, label="corrupt", color="#d62728", linewidth=2.0)
    ax2.plot(
        x,
        clean_comp_prob,
        label=f"clean {competitor_label}",
        color="#1f77b4",
        linewidth=1.4,
        linestyle="--",
        alpha=0.85,
    )
    ax2.plot(
        x,
        corr_comp_prob,
        label=f"corrupt {competitor_label}",
        color="#d62728",
        linewidth=1.4,
        linestyle="--",
        alpha=0.85,
    )
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Probability")
    ax2.set_title("<tool_call> and Competitor Probability by Layer")
    ax2.legend(frameon=False, ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_circuit_graph(
    circuit_df: pd.DataFrame,
    out_path: Path,
    topk: int = 60,
) -> None:
    # Keep top edges for readability.
    df = circuit_df[circuit_df["keep"] == 1].copy()
    if len(df) > topk:
        df = df.nlargest(topk, "DeltaS")

    g = nx.DiGraph()
    g.add_node("Input")
    g.add_node("Output")

    for _, row in df.iterrows():
        if row["type"] == "head":
            node = f"L{int(row['layer'])}H{int(row['head'])}"
        else:
            node = f"MLP{int(row['layer'])}"
        g.add_node(node, layer=int(row["layer"]))
        g.add_edge("Input", node, weight=0.2)
        g.add_edge(node, "Output", weight=float(row["DeltaS"]))

    # Layered layout.
    pos: Dict[str, Tuple[float, float]] = {"Input": (0.0, 0.5), "Output": (1.0, 0.5)}
    by_layer: Dict[int, List[str]] = {}
    for n, data in g.nodes(data=True):
        if n in {"Input", "Output"}:
            continue
        by_layer.setdefault(int(data.get("layer", 0)), []).append(n)

    max_layer = max(by_layer.keys()) if by_layer else 1
    for l, nodes in by_layer.items():
        nodes = sorted(nodes)
        ys = np.linspace(0.08, 0.92, num=len(nodes))
        x = (l + 1) / (max_layer + 2)
        for n, y in zip(nodes, ys):
            pos[n] = (x, float(y))

    fig, ax = plt.subplots(figsize=(13, 7))

    node_colors = []
    for n in g.nodes():
        if n == "Input":
            node_colors.append("#666666")
        elif n == "Output":
            node_colors.append("#222222")
        elif n.startswith("MLP"):
            node_colors.append("#1f77b4")
        else:
            node_colors.append("#d62728")

    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=380, ax=ax, alpha=0.95)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)

    input_edges = [(u, v) for u, v in g.edges() if u == "Input"]
    out_edges = [(u, v) for u, v in g.edges() if v == "Output"]
    nx.draw_networkx_edges(g, pos, edgelist=input_edges, width=0.8, edge_color="#aaaaaa", alpha=0.7, ax=ax)

    if out_edges:
        ws = np.array([g[u][v]["weight"] for u, v in out_edges], dtype=float)
        w_norm = (ws - ws.min()) / (ws.max() - ws.min() + 1e-9)
        widths = 0.8 + 2.8 * w_norm
        colors = plt.cm.RdBu_r((ws - ws.min()) / (ws.max() - ws.min() + 1e-9))
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=out_edges,
            width=widths,
            edge_color=colors,
            ax=ax,
            arrows=True,
            arrowsize=10,
            alpha=0.95,
        )

    ax.set_title("Discovered Tool-Call Circuit (Top Edges)")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


@torch.no_grad()
def probe_key_head(
    model,
    tokenizer,
    layer_idx: int,
    head_idx: int,
    sample_clean_text: str,
    tool_id: int,
    out_path: Path,
    device: torch.device,
) -> Dict[str, float]:
    tok = tokenizer(sample_clean_text, return_tensors="pt", add_special_tokens=False)
    tok = {k: v.to(device) for k, v in tok.items()}
    input_ids = tok["input_ids"]
    pos = input_ids.shape[1] - 1
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    # 1) attention pattern
    out = model_forward(model, **tok, output_attentions=True)
    attn = out.attentions[layer_idx][0, head_idx, pos].float().cpu().numpy()

    # 2) head write-to-vocab approximation
    holder = {}

    def capture(mod, inp):
        holder["pre"] = inp[0][0, pos].detach().float().cpu()

    h = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(capture)
    _ = model_forward(model, **tok)
    h.remove()

    pre = holder["pre"]
    s, e = head_idx * head_dim, (head_idx + 1) * head_dim
    head_slice = pre[s:e].to(device=device, dtype=model.dtype)
    w = model.model.layers[layer_idx].self_attn.o_proj.weight[:, s:e]
    contrib = torch.matmul(head_slice, w.T).float()  # [hidden]
    topk = 10
    # Chunked unembedding to avoid allocating full vocab score vector on GPU.
    lm_head_w = model.lm_head.weight
    best_vals = torch.full((topk,), -1e30, device=device, dtype=torch.float32)
    best_ids = torch.full((topk,), 0, device=device, dtype=torch.long)
    chunk = 4096
    for st in range(0, lm_head_w.shape[0], chunk):
        ed = min(st + chunk, lm_head_w.shape[0])
        scores = torch.matmul(lm_head_w[st:ed].float(), contrib)  # [chunk_vocab]
        cur_k = min(topk, scores.shape[0])
        cur_vals, cur_idx = torch.topk(scores, k=cur_k)
        cur_ids = cur_idx + st
        merged_vals = torch.cat([best_vals, cur_vals], dim=0)
        merged_ids = torch.cat([best_ids, cur_ids], dim=0)
        keep = torch.topk(merged_vals, k=topk)
        best_vals = keep.values
        best_ids = merged_ids[keep.indices]

    vals = best_vals
    ids = best_ids
    top_tokens = [token_name(tokenizer, int(i.item())) for i in ids]
    top_vals = vals.float().cpu().numpy()

    # 3) single-point ablation impact
    logits_base = out.logits[0, -1].float()
    prob_base = float(torch.softmax(logits_base, dim=-1)[tool_id].item())
    rank_base = int((logits_base > logits_base[tool_id]).sum().item()) + 1

    def ablate(mod, inp):
        x = inp[0].clone()
        x[pos, s:e] = 0.0
        return (x.unsqueeze(0),) if x.dim() == 2 else (x,)

    # Adjust for batched shape [1,S,H]
    def ablate_batched(mod, inp):
        x = inp[0].clone()
        x[0, pos, s:e] = 0.0
        return (x,)

    h2 = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(ablate_batched)
    logits_ab = model_forward(model, **tok).logits[0, -1].float()
    h2.remove()

    prob_ab = float(torch.softmax(logits_ab, dim=-1)[tool_id].item())
    rank_ab = int((logits_ab > logits_ab[tool_id]).sum().item()) + 1

    tokens = [token_name(tokenizer, int(t)) for t in input_ids[0].tolist()]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(np.arange(len(attn)), attn, color="#d62728", linewidth=1.8)
    ax1.set_title(f"Attention Pattern at p* (L{layer_idx}H{head_idx})")
    ax1.set_xlabel("Token position")
    ax1.set_ylabel("Attention weight")
    ax1.set_xlim(0, len(attn) - 1)

    # Mark top attended tokens
    top_attn_idx = np.argsort(attn)[-5:][::-1]
    ann = [f"{i}:{tokens[i][:12]}" for i in top_attn_idx]
    ax1.text(
        0.01,
        0.98,
        "Top attended: " + " | ".join(ann),
        transform=ax1.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ax2 = fig.add_subplot(gs[1, 0])
    y = np.arange(topk)
    ax2.barh(y, top_vals, color="#1f77b4")
    ax2.set_yticks(y)
    ax2.set_yticklabels(top_tokens)
    ax2.invert_yaxis()
    ax2.set_title("Head Write Top-K Token Scores")
    ax2.set_xlabel("Unembed score")

    ax3 = fig.add_subplot(gs[1, 1])
    xs = np.arange(2)
    ax3.bar(xs - 0.15, [prob_base, prob_ab], width=0.3, label="P(<tool_call>)", color="#2ca02c")
    ax3_2 = ax3.twinx()
    ax3_2.bar(xs + 0.15, [rank_base, rank_ab], width=0.3, label="Rank(<tool_call>)", color="#ff7f0e")
    ax3.set_xticks(xs)
    ax3.set_xticklabels(["baseline", "ablate"])
    ax3.set_title("Single-Head Ablation Effect")
    ax3.set_ylabel("Probability")
    ax3_2.set_ylabel("Rank")

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_2.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "prob_base": prob_base,
        "prob_ablate": prob_ab,
        "rank_base": rank_base,
        "rank_ablate": rank_ab,
    }


@torch.no_grad()
def single_head_ablation_effect(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    head_idx: int,
    tool_id: int,
    device: torch.device,
) -> Dict[str, float]:
    tok = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    tok = {k: v.to(device) for k, v in tok.items()}
    pos = tok["input_ids"].shape[1] - 1
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    s = head_idx * head_dim
    e = (head_idx + 1) * head_dim

    base_logits = model_forward(model, **tok).logits[0, -1].float()
    prob_base = float(torch.softmax(base_logits, dim=-1)[tool_id].item())
    rank_base = int((base_logits > base_logits[tool_id]).sum().item()) + 1

    def ablate_hook(mod, inp):
        x = inp[0].clone()
        x[0, pos, s:e] = 0.0
        return (x,)

    h = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(ablate_hook)
    ab_logits = model_forward(model, **tok).logits[0, -1].float()
    h.remove()

    prob_ab = float(torch.softmax(ab_logits, dim=-1)[tool_id].item())
    rank_ab = int((ab_logits > ab_logits[tool_id]).sum().item()) + 1
    return {
        "prob_base": prob_base,
        "prob_ablate": prob_ab,
        "rank_base": rank_base,
        "rank_ablate": rank_ab,
        "prob_drop": prob_base - prob_ab,
        "rank_increase": rank_ab - rank_base,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    maybe_disable_hf_progress()
    setup_plot_style()

    pair_dir = Path(args.pair_dir)
    report_dir = Path(args.report_dir)
    fig_dir = Path(args.fig_dir)
    case_dir = Path(args.case_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()
    gpu_before = query_gpu()

    dtype = dtype_from_str(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map="cuda" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    tool_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    if tool_id is None or tool_id < 0:
        ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError("<tool_call> is not a single token; current script assumes single token.")
        tool_id = ids[0]

    examples = load_examples(pair_dir)
    if len(examples) == 0:
        raise RuntimeError("No meta-q*.json found in pair dir")

    # A0 baseline recompute
    baseline_df = build_baseline_table(
        examples,
        model,
        tokenizer,
        device,
        tool_id,
        batch_size=args.batch_size_baseline,
    )
    baseline_path = report_dir / "baseline_first_token.csv"
    baseline_df.to_csv(baseline_path, index=False)

    clean_rate = baseline_df["clean_is_tool"].mean()
    corr_rate = baseline_df["corr_is_tool"].mean()
    balanced_rate = baseline_df["balanced"].mean()

    # Samples for heavy analysis: balanced set only.
    balanced_q = baseline_df.loc[baseline_df["balanced"] == 1, "q"].tolist()
    balanced_q_set = set(balanced_q)
    balanced_examples = [x for x in examples if x.qid in balanced_q_set]

    if len(balanced_examples) < args.min_analysis_pairs:
        analysis_examples = balanced_examples
    else:
        n_take = min(args.analysis_max_pairs, len(balanced_examples))
        analysis_examples = balanced_examples[:n_take]
    analysis_q_set = {x.qid for x in analysis_examples}

    holdout_examples = [x for x in balanced_examples if x.qid not in analysis_q_set]
    if args.completeness_test_max_pairs > 0:
        holdout_examples = holdout_examples[: args.completeness_test_max_pairs]
    reuse_val_for_test = len(holdout_examples) == 0
    if reuse_val_for_test:
        holdout_examples = analysis_examples

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    hidden = model.config.hidden_size
    head_dim = hidden // n_heads

    # Accumulators over analysis set.
    ap_head = np.zeros((n_layers, n_heads), dtype=np.float64)
    ct_head = np.zeros((n_layers, n_heads), dtype=np.float64)
    s_clean_head = np.zeros((n_layers, n_heads), dtype=np.float64)
    s_corr_head = np.zeros((n_layers, n_heads), dtype=np.float64)

    ap_mlp = np.zeros((n_layers,), dtype=np.float64)
    ct_mlp = np.zeros((n_layers,), dtype=np.float64)
    s_clean_mlp = np.zeros((n_layers,), dtype=np.float64)
    s_corr_mlp = np.zeros((n_layers,), dtype=np.float64)

    # Main AP/CT + ablation loop
    per_sample_log = []
    torch.cuda.reset_peak_memory_stats(device=device) if device.type == "cuda" else None

    for idx, ex in enumerate(analysis_examples):
        clean_tok = tokenizer(ex.clean_text, return_tensors="pt", add_special_tokens=False)
        corr_tok = tokenizer(ex.corrupt_text, return_tensors="pt", add_special_tokens=False)

        clean_ids = clean_tok["input_ids"].to(device)
        clean_mask = clean_tok["attention_mask"].to(device)
        corr_ids = corr_tok["input_ids"].to(device)
        corr_mask = corr_tok["attention_mask"].to(device)

        if clean_ids.shape[1] != corr_ids.shape[1]:
            # Alignment should hold per data design. Skip if it doesn't.
            continue

        pos = clean_ids.shape[1] - 1

        attn_clean, mlp_clean = collect_clean_cache(model, clean_ids, clean_mask, pos, n_layers)
        corr_margin, corr_logp, _ = run_corr_baseline(model, corr_ids, corr_mask, tool_id)
        clean_logp, _ = run_clean_baseline(model, clean_ids, clean_mask, tool_id)

        for l in range(n_layers):
            ap_head[l] += head_batch_ap(
                model,
                corr_ids,
                corr_mask,
                pos,
                l,
                attn_clean[l],
                n_heads,
                head_dim,
                args.head_batch_size,
                tool_id,
                corr_margin,
            )
            ct_head[l] += head_batch_ct(
                model,
                corr_ids,
                corr_mask,
                pos,
                l,
                attn_clean[l],
                n_heads,
                head_dim,
                args.head_batch_size,
                tool_id,
                corr_margin,
            )
            s_corr_head[l] += head_batch_ablate_logprob_delta(
                model,
                corr_ids,
                corr_mask,
                pos,
                l,
                n_heads,
                head_dim,
                args.head_batch_size,
                tool_id,
                corr_logp,
            )
            s_clean_head[l] += head_batch_ablate_logprob_delta(
                model,
                clean_ids,
                clean_mask,
                pos,
                l,
                n_heads,
                head_dim,
                args.head_batch_size,
                tool_id,
                clean_logp,
            )

            mlp_delta = mlp_patch_margin_delta(
                model,
                corr_ids,
                corr_mask,
                pos,
                l,
                mlp_clean[l],
                tool_id,
                corr_margin,
            )
            ap_mlp[l] += mlp_delta
            # CT(MLP edge) coincides with patching its direct residual write.
            ct_mlp[l] += mlp_delta
            s_corr_mlp[l] += mlp_ablate_logprob_delta(
                model,
                corr_ids,
                corr_mask,
                pos,
                l,
                tool_id,
                corr_logp,
            )
            s_clean_mlp[l] += mlp_ablate_logprob_delta(
                model,
                clean_ids,
                clean_mask,
                pos,
                l,
                tool_id,
                clean_logp,
            )

        per_sample_log.append(
            {
                "q": ex.qid,
                "seq_len": int(clean_ids.shape[1]),
                "corr_margin": corr_margin,
                "corr_logprob_tool": corr_logp,
                "clean_logprob_tool": clean_logp,
                "sample_index": idx,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_eff = max(len(per_sample_log), 1)
    ap_head /= n_eff
    ct_head /= n_eff
    s_clean_head /= n_eff
    s_corr_head /= n_eff

    ap_mlp /= n_eff
    ct_mlp /= n_eff
    s_clean_mlp /= n_eff
    s_corr_mlp /= n_eff

    # A2 figures
    ap_vmin, ap_vmax = plot_heatmap(
        ap_head,
        "AP Head Contribution (margin delta, corrupt patched by clean)",
        fig_dir / "ap_head_heatmap.png",
        center_zero=True,
        cmap="RdBu_r",
    )
    ct_vmin, ct_vmax = plot_heatmap(
        ct_head,
        "CT Head Contribution (edge patch, margin delta)",
        fig_dir / "ct_head_heatmap.png",
        center_zero=True,
        cmap="RdBu_r",
    )

    # Build per-pair activation caches for completeness evaluation (D_val / D_test split).
    val_pair_caches = build_pair_activation_caches(
        model,
        tokenizer,
        analysis_examples,
        n_layers=n_layers,
        device=device,
    )
    if len(val_pair_caches) == 0:
        raise RuntimeError("No aligned clean/corrupt pairs available for D_val completeness evaluation.")

    test_pair_caches = build_pair_activation_caches(
        model,
        tokenizer,
        holdout_examples,
        n_layers=n_layers,
        device=device,
    )
    if len(test_pair_caches) == 0:
        test_pair_caches = val_pair_caches
        reuse_val_for_test = True

    val_qids = [x.qid for x in val_pair_caches]
    test_qids = [x.qid for x in test_pair_caches]
    eval_qids = sorted(set(val_qids).union(set(test_qids)))
    val_slice = baseline_df[baseline_df["q"].isin(val_qids)]
    test_slice = baseline_df[baseline_df["q"].isin(test_qids)]

    val_full_metrics, val_full_ci, val_n = metrics_from_baseline_slice(val_slice)
    test_full_metrics, test_full_ci, test_n = metrics_from_baseline_slice(test_slice)

    completeness_rows = []
    completeness_summary_rows = []
    best_tau = args.tau_grid[0]
    best_objective = -1e9
    best_keep_head = None
    best_keep_mlp = None
    best_circuit_df = None
    best_random_mean = None
    total_components = n_layers * n_heads + n_layers

    for tau in args.tau_grid:
        df_tau = build_circuit_df(
            ap_head,
            ct_head,
            s_clean_head,
            s_corr_head,
            ap_mlp,
            ct_mlp,
            s_clean_mlp,
            s_corr_mlp,
            tau=tau,
        )
        selected_idx = select_bidirectional_indices(df_tau, tau=tau, max_edges=args.max_circuit_edges)
        df_tau["keep"] = 0
        df_tau.loc[selected_idx, "keep"] = 1

        keep_head = np.zeros((n_layers, n_heads), dtype=bool)
        keep_mlp = np.zeros((n_layers,), dtype=bool)
        for _, row in df_tau[df_tau["keep"] == 1].iterrows():
            if row["type"] == "head":
                keep_head[int(row["layer"]), int(row["head"])] = True
            else:
                keep_mlp[int(row["layer"])] = True

        n_keep_heads = int(keep_head.sum())
        n_keep_mlps = int(keep_mlp.sum())
        n_keep_total = n_keep_heads + n_keep_mlps
        edge_ratio = n_keep_total / max(total_components, 1)
        head_ratio = n_keep_heads / max(n_layers * n_heads, 1)
        mlp_ratio = n_keep_mlps / max(n_layers, 1)

        circ_val_detail = evaluate_with_circuit_detailed(
            model,
            val_pair_caches,
            keep_head,
            keep_mlp,
            tool_id,
            device,
            non_circuit_retain=args.non_circuit_retain,
        )
        circ_test_detail = evaluate_with_circuit_detailed(
            model,
            test_pair_caches,
            keep_head,
            keep_mlp,
            tool_id,
            device,
            non_circuit_retain=args.non_circuit_retain,
        )
        circ_val_metrics, circ_val_ci, _ = metrics_from_detailed_eval(circ_val_detail)
        circ_test_metrics, circ_test_ci, _ = metrics_from_detailed_eval(circ_test_detail)

        rnd_val_clean_trials: List[float] = []
        rnd_val_rej_trials: List[float] = []
        rnd_val_bal_trials: List[float] = []
        rnd_test_clean_trials: List[float] = []
        rnd_test_rej_trials: List[float] = []
        rnd_test_bal_trials: List[float] = []

        for t in range(args.random_trials):
            rnd_keep_head, rnd_keep_mlp = make_random_circuit(
                n_layers=n_layers,
                n_heads=n_heads,
                n_keep_heads=n_keep_heads,
                n_keep_mlps=n_keep_mlps,
                seed=args.seed + int(tau * 1000) + t,
            )
            rnd_val = evaluate_with_circuit(
                model,
                val_pair_caches,
                rnd_keep_head,
                rnd_keep_mlp,
                tool_id,
                device,
                non_circuit_retain=args.non_circuit_retain,
            )
            rnd_test = evaluate_with_circuit(
                model,
                test_pair_caches,
                rnd_keep_head,
                rnd_keep_mlp,
                tool_id,
                device,
                non_circuit_retain=args.non_circuit_retain,
            )
            rnd_val_clean_trials.append(float(rnd_val["ToolCall@1_clean"]))
            rnd_val_rej_trials.append(float(rnd_val["Reject@1_corr"]))
            rnd_val_bal_trials.append(float(rnd_val["Balanced"]))
            rnd_test_clean_trials.append(float(rnd_test["ToolCall@1_clean"]))
            rnd_test_rej_trials.append(float(rnd_test["Reject@1_corr"]))
            rnd_test_bal_trials.append(float(rnd_test["Balanced"]))

        rnd_val_clean_trials_arr = np.array(rnd_val_clean_trials, dtype=float)
        rnd_val_rej_trials_arr = np.array(rnd_val_rej_trials, dtype=float)
        rnd_val_bal_trials_arr = np.array(rnd_val_bal_trials, dtype=float)
        rnd_test_clean_trials_arr = np.array(rnd_test_clean_trials, dtype=float)
        rnd_test_rej_trials_arr = np.array(rnd_test_rej_trials, dtype=float)
        rnd_test_bal_trials_arr = np.array(rnd_test_bal_trials, dtype=float)

        rnd_val_metrics = {
            "ToolCall@1_clean": float(rnd_val_clean_trials_arr.mean()),
            "Reject@1_corr": float(rnd_val_rej_trials_arr.mean()),
            "Balanced": float(rnd_val_bal_trials_arr.mean()),
        }
        rnd_test_metrics = {
            "ToolCall@1_clean": float(rnd_test_clean_trials_arr.mean()),
            "Reject@1_corr": float(rnd_test_rej_trials_arr.mean()),
            "Balanced": float(rnd_test_bal_trials_arr.mean()),
        }
        rnd_val_std = {
            "ToolCall@1_clean_std": float(rnd_val_clean_trials_arr.std()),
            "Reject@1_corr_std": float(rnd_val_rej_trials_arr.std()),
            "Balanced_std": float(rnd_val_bal_trials_arr.std()),
        }
        rnd_test_std = {
            "ToolCall@1_clean_std": float(rnd_test_clean_trials_arr.std()),
            "Reject@1_corr_std": float(rnd_test_rej_trials_arr.std()),
            "Balanced_std": float(rnd_test_bal_trials_arr.std()),
        }
        rnd_val_ci = {
            "ToolCall@1_clean_ci": (
                float(np.percentile(rnd_val_clean_trials_arr, 2.5)),
                float(np.percentile(rnd_val_clean_trials_arr, 97.5)),
            ),
            "Reject@1_corr_ci": (
                float(np.percentile(rnd_val_rej_trials_arr, 2.5)),
                float(np.percentile(rnd_val_rej_trials_arr, 97.5)),
            ),
            "Balanced_ci": (
                float(np.percentile(rnd_val_bal_trials_arr, 2.5)),
                float(np.percentile(rnd_val_bal_trials_arr, 97.5)),
            ),
        }
        rnd_test_ci = {
            "ToolCall@1_clean_ci": (
                float(np.percentile(rnd_test_clean_trials_arr, 2.5)),
                float(np.percentile(rnd_test_clean_trials_arr, 97.5)),
            ),
            "Reject@1_corr_ci": (
                float(np.percentile(rnd_test_rej_trials_arr, 2.5)),
                float(np.percentile(rnd_test_rej_trials_arr, 97.5)),
            ),
            "Balanced_ci": (
                float(np.percentile(rnd_test_bal_trials_arr, 2.5)),
                float(np.percentile(rnd_test_bal_trials_arr, 97.5)),
            ),
        }

        circ_val_score = (
            circ_val_metrics["Balanced"]
            + 0.2 * circ_val_metrics["ToolCall@1_clean"]
            + 0.2 * circ_val_metrics["Reject@1_corr"]
        )
        circ_test_score = (
            circ_test_metrics["Balanced"]
            + 0.2 * circ_test_metrics["ToolCall@1_clean"]
            + 0.2 * circ_test_metrics["Reject@1_corr"]
        )
        rnd_val_score_trials = (
            rnd_val_bal_trials_arr + 0.2 * rnd_val_clean_trials_arr + 0.2 * rnd_val_rej_trials_arr
        )
        rnd_test_score_trials = (
            rnd_test_bal_trials_arr + 0.2 * rnd_test_clean_trials_arr + 0.2 * rnd_test_rej_trials_arr
        )
        objective_val = float(circ_val_score - rnd_val_score_trials.mean())
        objective_test = float(circ_test_score - rnd_test_score_trials.mean())

        p_val_balanced_vs_random = float(
            (1 + np.sum(rnd_val_bal_trials_arr >= circ_val_metrics["Balanced"]))
            / (len(rnd_val_bal_trials_arr) + 1)
        )
        p_val_objective_vs_random = float(
            (1 + np.sum(rnd_val_score_trials >= circ_val_score)) / (len(rnd_val_score_trials) + 1)
        )
        p_test_balanced_vs_random = float(
            (1 + np.sum(rnd_test_bal_trials_arr >= circ_test_metrics["Balanced"]))
            / (len(rnd_test_bal_trials_arr) + 1)
        )
        p_test_objective_vs_random = float(
            (1 + np.sum(rnd_test_score_trials >= circ_test_score)) / (len(rnd_test_score_trials) + 1)
        )

        z_test_balanced = float(
            (circ_test_metrics["Balanced"] - rnd_test_metrics["Balanced"])
            / max(rnd_test_std["Balanced_std"], 1e-9)
        )

        split_specs = [
            ("D_val", val_full_metrics, val_full_ci, val_n, circ_val_metrics, circ_val_ci, rnd_val_metrics, rnd_val_ci, rnd_val_std, objective_val, p_val_balanced_vs_random, p_val_objective_vs_random),
            ("D_test", test_full_metrics, test_full_ci, test_n, circ_test_metrics, circ_test_ci, rnd_test_metrics, rnd_test_ci, rnd_test_std, objective_test, p_test_balanced_vs_random, p_test_objective_vs_random),
        ]
        for (
            split_name,
            full_metrics,
            full_ci,
            n_split,
            circ_metrics,
            circ_ci,
            rnd_metrics,
            rnd_ci,
            rnd_std,
            objective_split,
            p_balanced_split,
            p_objective_split,
        ) in split_specs:
            completeness_rows.extend(
                [
                    {
                        "tau": tau,
                        "split": split_name,
                        "setting": "Full model",
                        **full_metrics,
                        "ToolCall@1_clean_ci_low": full_ci["ToolCall@1_clean_ci"][0],
                        "ToolCall@1_clean_ci_high": full_ci["ToolCall@1_clean_ci"][1],
                        "Reject@1_corr_ci_low": full_ci["Reject@1_corr_ci"][0],
                        "Reject@1_corr_ci_high": full_ci["Reject@1_corr_ci"][1],
                        "Balanced_ci_low": full_ci["Balanced_ci"][0],
                        "Balanced_ci_high": full_ci["Balanced_ci"][1],
                        "n_samples": n_split,
                        "filter": "balanced(clean tool, corr non-tool)",
                        "n_keep_heads": n_layers * n_heads,
                        "n_keep_mlps": n_layers,
                        "n_keep_total": total_components,
                        "edge_ratio": 1.0,
                        "head_ratio": 1.0,
                        "mlp_ratio": 1.0,
                        "non_circuit_retain": args.non_circuit_retain,
                        "random_trials": args.random_trials,
                        "objective_gap_vs_random": np.nan,
                        "delta_balanced_vs_random": np.nan,
                        "delta_clean_vs_random": np.nan,
                        "delta_reject_vs_random": np.nan,
                        "p_balanced_vs_random": np.nan,
                        "p_objective_vs_random": np.nan,
                        "Balanced_std": np.nan,
                    },
                    {
                        "tau": tau,
                        "split": split_name,
                        "setting": "Circuit-only",
                        **circ_metrics,
                        "ToolCall@1_clean_ci_low": circ_ci["ToolCall@1_clean_ci"][0],
                        "ToolCall@1_clean_ci_high": circ_ci["ToolCall@1_clean_ci"][1],
                        "Reject@1_corr_ci_low": circ_ci["Reject@1_corr_ci"][0],
                        "Reject@1_corr_ci_high": circ_ci["Reject@1_corr_ci"][1],
                        "Balanced_ci_low": circ_ci["Balanced_ci"][0],
                        "Balanced_ci_high": circ_ci["Balanced_ci"][1],
                        "n_samples": n_split,
                        "filter": "balanced(clean tool, corr non-tool)",
                        "n_keep_heads": n_keep_heads,
                        "n_keep_mlps": n_keep_mlps,
                        "n_keep_total": n_keep_total,
                        "edge_ratio": edge_ratio,
                        "head_ratio": head_ratio,
                        "mlp_ratio": mlp_ratio,
                        "non_circuit_retain": args.non_circuit_retain,
                        "random_trials": args.random_trials,
                        "objective_gap_vs_random": objective_split,
                        "delta_balanced_vs_random": circ_metrics["Balanced"] - rnd_metrics["Balanced"],
                        "delta_clean_vs_random": circ_metrics["ToolCall@1_clean"] - rnd_metrics["ToolCall@1_clean"],
                        "delta_reject_vs_random": circ_metrics["Reject@1_corr"] - rnd_metrics["Reject@1_corr"],
                        "p_balanced_vs_random": p_balanced_split,
                        "p_objective_vs_random": p_objective_split,
                        "Balanced_std": np.nan,
                    },
                    {
                        "tau": tau,
                        "split": split_name,
                        "setting": "Random same-size",
                        **rnd_metrics,
                        "ToolCall@1_clean_ci_low": rnd_ci["ToolCall@1_clean_ci"][0],
                        "ToolCall@1_clean_ci_high": rnd_ci["ToolCall@1_clean_ci"][1],
                        "Reject@1_corr_ci_low": rnd_ci["Reject@1_corr_ci"][0],
                        "Reject@1_corr_ci_high": rnd_ci["Reject@1_corr_ci"][1],
                        "Balanced_ci_low": rnd_ci["Balanced_ci"][0],
                        "Balanced_ci_high": rnd_ci["Balanced_ci"][1],
                        "n_samples": n_split,
                        "filter": "balanced(clean tool, corr non-tool)",
                        "n_keep_heads": n_keep_heads,
                        "n_keep_mlps": n_keep_mlps,
                        "n_keep_total": n_keep_total,
                        "edge_ratio": edge_ratio,
                        "head_ratio": head_ratio,
                        "mlp_ratio": mlp_ratio,
                        "non_circuit_retain": args.non_circuit_retain,
                        "random_trials": args.random_trials,
                        "objective_gap_vs_random": np.nan,
                        "delta_balanced_vs_random": np.nan,
                        "delta_clean_vs_random": np.nan,
                        "delta_reject_vs_random": np.nan,
                        "p_balanced_vs_random": np.nan,
                        "p_objective_vs_random": np.nan,
                        "Balanced_std": rnd_std["Balanced_std"],
                        "ToolCall@1_clean_std": rnd_std["ToolCall@1_clean_std"],
                        "Reject@1_corr_std": rnd_std["Reject@1_corr_std"],
                    },
                ]
            )

        completeness_summary_rows.append(
            {
                "tau": tau,
                "n_keep_total": n_keep_total,
                "edge_ratio": edge_ratio,
                "head_ratio": head_ratio,
                "mlp_ratio": mlp_ratio,
                "D_val_Full_Balanced": val_full_metrics["Balanced"],
                "D_val_Circuit_Balanced": circ_val_metrics["Balanced"],
                "D_val_Random_Balanced": rnd_val_metrics["Balanced"],
                "D_test_Full_Balanced": test_full_metrics["Balanced"],
                "D_test_Circuit_Balanced": circ_test_metrics["Balanced"],
                "D_test_Random_Balanced": rnd_test_metrics["Balanced"],
                "D_test_Random_Balanced_std": rnd_test_std["Balanced_std"],
                "D_val_delta_Balanced": circ_val_metrics["Balanced"] - rnd_val_metrics["Balanced"],
                "D_test_delta_Balanced": circ_test_metrics["Balanced"] - rnd_test_metrics["Balanced"],
                "objective_gap_vs_random_D_val": objective_val,
                "objective_gap_vs_random_D_test": objective_test,
                "p_D_val_balanced_vs_random": p_val_balanced_vs_random,
                "p_D_val_objective_vs_random": p_val_objective_vs_random,
                "p_D_test_balanced_vs_random": p_test_balanced_vs_random,
                "p_D_test_objective_vs_random": p_test_objective_vs_random,
                "z_D_test_balanced_vs_random": z_test_balanced,
            }
        )

        if objective_val > best_objective:
            best_objective = objective_val
            best_tau = tau
            best_keep_head = keep_head.copy()
            best_keep_mlp = keep_mlp.copy()
            best_circuit_df = df_tau.copy()
            best_random_mean = rnd_test_metrics.copy()

    completeness_df = pd.DataFrame(completeness_rows)
    completeness_df.to_csv(report_dir / "completeness.csv", index=False)
    completeness_summary_df = pd.DataFrame(completeness_summary_rows)
    completeness_summary_df.to_csv(report_dir / "completeness_summary.csv", index=False)
    completeness_table_like_paper = completeness_summary_df[
        [
            "tau",
            "n_keep_total",
            "edge_ratio",
            "D_val_Full_Balanced",
            "D_val_Circuit_Balanced",
            "D_test_Full_Balanced",
            "D_test_Random_Balanced",
            "D_test_Circuit_Balanced",
            "D_test_delta_Balanced",
            "objective_gap_vs_random_D_test",
            "p_D_test_balanced_vs_random",
        ]
    ].copy()
    completeness_table_like_paper.rename(
        columns={
            "n_keep_total": "#Edge",
            "D_val_Full_Balanced": "D_val Original(G)",
            "D_val_Circuit_Balanced": "D_val Circuit(C)",
            "D_test_Full_Balanced": "D_test Original(G)",
            "D_test_Random_Balanced": "D_test Random",
            "D_test_Circuit_Balanced": "D_test Circuit(C)",
        },
        inplace=True,
    )
    completeness_table_like_paper.to_csv(report_dir / "completeness_table_like_paper.csv", index=False)

    if best_circuit_df is None or best_keep_head is None or best_keep_mlp is None:
        raise RuntimeError("Failed to build circuit across tau grid")
    if best_random_mean is None:
        raise RuntimeError("Failed to build random baseline for best tau")

    primary_split = "D_test" if not reuse_val_for_test else "D_val"
    best_circ_row = completeness_df[
        (completeness_df["tau"] == best_tau)
        & (completeness_df["split"] == primary_split)
        & (completeness_df["setting"] == "Circuit-only")
    ].iloc[0]
    best_rnd_row = completeness_df[
        (completeness_df["tau"] == best_tau)
        & (completeness_df["split"] == primary_split)
        & (completeness_df["setting"] == "Random same-size")
    ].iloc[0]
    best_full_row = completeness_df[
        (completeness_df["tau"] == best_tau)
        & (completeness_df["split"] == primary_split)
        & (completeness_df["setting"] == "Full model")
    ].iloc[0]
    best_summary_row = completeness_summary_df[completeness_summary_df["tau"] == best_tau].iloc[0]

    # Save final circuit edge report
    best_circuit_df.to_csv(report_dir / "circuit_edges.csv", index=False)

    # A4 layer distribution + rank/prob by layer
    plot_layer_distribution(best_keep_head, best_keep_mlp, fig_dir / "layer_distribution.png")

    eval_slice = baseline_df[baseline_df["q"].isin(eval_qids)]
    corr_non_tool = eval_slice[eval_slice["corr_is_tool"] == 0]
    if len(corr_non_tool) > 0:
        competitor_id = int(corr_non_tool["corr_top1_id"].value_counts().idxmax())
        competitor_label = token_name(tokenizer, competitor_id)
    else:
        competitor_id = None
        competitor_label = "competitor"

    clean_rank, clean_prob, corr_rank, corr_prob, clean_comp_prob, corr_comp_prob = layer_rank_prob(
        model,
        tokenizer,
        analysis_examples,
        tool_id,
        competitor_id,
        device,
    )
    plot_rank_prob(
        clean_rank,
        clean_prob,
        corr_rank,
        corr_prob,
        clean_comp_prob,
        corr_comp_prob,
        competitor_label=competitor_label,
        out_path=fig_dir / "rank_prob_by_layer.png",
    )

    # A1 final circuit graph
    plot_circuit_graph(best_circuit_df, fig_dir / "final_circuit.png")

    case_meta = case_dir / "meta-q85.json"
    case_clean_path = case_dir / "prompt-clean-q85.txt"
    if case_meta.exists() and case_clean_path.exists():
        case_clean = case_clean_path.read_text()
    else:
        # fallback to project-local sample
        case_clean = (Path("sample") / "prompt-clean-q85.txt").read_text()

    # A3 key head probe: choose a positive-support head and rerank by direct ablation effect.
    head_df = best_circuit_df[(best_circuit_df["type"] == "head") & (best_circuit_df["keep"] == 1)].copy()
    if len(head_df) == 0:
        # fallback choose best head by DeltaS globally
        head_df = best_circuit_df[best_circuit_df["type"] == "head"].copy()
    probe_candidates = head_df[head_df["DeltaS"] > 0].copy()
    if len(probe_candidates) == 0:
        probe_candidates = head_df.copy()
    probe_candidates = probe_candidates.sort_values("DeltaS", ascending=False)

    # Rerank a broader positive pool by direct ablation on the running example.
    select_pool = probe_candidates.head(min(32, len(probe_candidates))).copy()
    best_idx = int(select_pool.index[0])
    best_drop = -1e9
    for idx, row in select_pool.iterrows():
        effect = single_head_ablation_effect(
            model,
            tokenizer,
            case_clean,
            int(row["layer"]),
            int(row["head"]),
            tool_id,
            device,
        )
        select_pool.loc[idx, "ablation_prob_drop"] = effect["prob_drop"]
        select_pool.loc[idx, "ablation_rank_increase"] = effect["rank_increase"]
        if effect["prob_drop"] > best_drop:
            best_drop = effect["prob_drop"]
            best_idx = int(idx)

    # If all candidate drops are weak/negative, fall back to score-based selection.
    if best_drop <= 0:
        best_idx = int(probe_candidates.index[0])

    top_head = probe_candidates.loc[best_idx]
    top_layer = int(top_head["layer"])
    top_hid = int(top_head["head"])

    probe_path = fig_dir / f"L{top_layer}H{top_hid}_probe.png"
    # Probe needs attention maps; SDPA does not expose them, so reload in eager mode.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    probe_device = device
    probe_error = None
    model_probe = None
    try:
        model_probe = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            device_map="cuda" if device.type == "cuda" else None,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        if device.type != "cuda":
            model_probe = model_probe.to(device)
        model_probe.eval()
        probe_stats = probe_key_head(
            model_probe,
            tokenizer,
            top_layer,
            top_hid,
            case_clean,
            tool_id,
            probe_path,
            probe_device,
        )
    except Exception as e:
        probe_error = str(e)
        if model_probe is not None:
            del model_probe
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Fallback to CPU probing to avoid losing full-run outputs on transient GPU pressure.
        probe_device = torch.device("cpu")
        model_probe = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(probe_device)
        model_probe.eval()
        probe_stats = probe_key_head(
            model_probe,
            tokenizer,
            top_layer,
            top_hid,
            case_clean,
            tool_id,
            probe_path,
            probe_device,
        )
    finally:
        if model_probe is not None:
            del model_probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Sanity checks and auto-adjust notes
    sanity = {
        "ap_top10_mean": float(np.mean(np.sort(ap_head.reshape(-1))[-10:])),
        "ct_top10_mean": float(np.mean(np.sort(ct_head.reshape(-1))[-10:])),
        "deltaS_top10_mean": float(
            np.mean(
                np.sort((s_clean_head - s_corr_head).reshape(-1))[-10:]
            )
        ),
        "best_tau": float(best_tau),
        "best_objective_gap_vs_random": float(best_objective),
        "best_balanced_circuit": float(best_circ_row["Balanced"]),
        "best_balanced_random": float(best_rnd_row["Balanced"]),
        "best_keep_heads": int(best_keep_head.sum()),
        "best_keep_mlps": int(best_keep_mlp.sum()),
        "best_edge_ratio": float(best_circ_row["edge_ratio"]),
        "primary_completeness_split": primary_split,
        "best_p_balanced_vs_random": float(best_circ_row["p_balanced_vs_random"]),
        "best_p_objective_vs_random": float(best_circ_row["p_objective_vs_random"]),
        "candidate_ablation_best_drop": float(best_drop),
        "probe_prob_drop": float(probe_stats["prob_base"] - probe_stats["prob_ablate"]),
        "probe_rank_increase": float(probe_stats["rank_ablate"] - probe_stats["rank_base"]),
        "probe_device": str(probe_device),
    }
    if probe_error is not None:
        sanity["probe_gpu_error"] = probe_error

    gpu_after = query_gpu()
    peak_mem_gb = (
        float(torch.cuda.max_memory_allocated(device=device) / (1024**3)) if device.type == "cuda" else 0.0
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "tool_token_id": int(tool_id),
        "n_total_pairs": len(examples),
        "n_balanced_pairs": len(balanced_examples),
        "n_analysis_pairs": len(analysis_examples),
        "n_completeness_pairs_d_val": len(val_pair_caches),
        "n_completeness_pairs_d_test": len(test_pair_caches),
        "d_test_reuses_d_val": bool(reuse_val_for_test),
        "completeness_primary_split": primary_split,
        "baseline": {
            "ToolCall@1_clean": float(clean_rate),
            "ToolCall@1_corr": float(corr_rate),
            "Reject@1_corr": float(1.0 - corr_rate),
            "Balanced": float(balanced_rate),
        },
        "best_tau": float(best_tau),
        "best_completeness": {
            "full_model": {
                "ToolCall@1_clean": float(best_full_row["ToolCall@1_clean"]),
                "Reject@1_corr": float(best_full_row["Reject@1_corr"]),
                "Balanced": float(best_full_row["Balanced"]),
            },
            "circuit": {
                "ToolCall@1_clean": float(best_circ_row["ToolCall@1_clean"]),
                "Reject@1_corr": float(best_circ_row["Reject@1_corr"]),
                "Balanced": float(best_circ_row["Balanced"]),
            },
            "random_same_size": {
                "ToolCall@1_clean": float(best_rnd_row["ToolCall@1_clean"]),
                "Reject@1_corr": float(best_rnd_row["Reject@1_corr"]),
                "Balanced": float(best_rnd_row["Balanced"]),
            },
            "split": primary_split,
            "edge_ratio": float(best_circ_row["edge_ratio"]),
            "delta_balanced_vs_random": float(best_circ_row["delta_balanced_vs_random"]),
            "p_balanced_vs_random": float(best_circ_row["p_balanced_vs_random"]),
            "p_objective_vs_random": float(best_circ_row["p_objective_vs_random"]),
            "objective_gap_vs_random": float(best_circ_row["objective_gap_vs_random"]),
            "D_val_delta_balanced_vs_random": float(best_summary_row["D_val_delta_Balanced"]),
            "D_test_delta_balanced_vs_random": float(best_summary_row["D_test_delta_Balanced"]),
        },
        "top_probe_head": f"L{top_layer}H{top_hid}",
        "probe": probe_stats,
        "sanity": sanity,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "dtype": args.dtype,
        "batch_size_baseline": args.batch_size_baseline,
        "non_circuit_retain": args.non_circuit_retain,
        "max_circuit_edges": int(args.max_circuit_edges),
        "random_trials": int(args.random_trials),
        "peak_cuda_mem_gb": peak_mem_gb,
        "elapsed_sec": time.time() - start_ts,
    }

    (report_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (report_dir / "analysis_sample_log.json").write_text(
        json.dumps(per_sample_log, indent=2, ensure_ascii=False)
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
