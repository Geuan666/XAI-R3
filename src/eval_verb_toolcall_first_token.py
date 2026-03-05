#!/usr/bin/env python3
"""Evaluate first-token <tool_call> activation under verb replacements."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_VERBS = [
    "Write",
    "State",
    "Provide",
    "Give",
    "Produce",
    "Generate",
    "Create",
    "Draft",
    "Compose",
    "Return",
    "Output",
    "Present",
    "Deliver",
    "Offer",
    "Supply",
    "Formulate",
    "Construct",
    "Build",
    "Develop",
    "Implement",
    "Code",
    "Program",
    "Solve",
    "Complete",
    "Finish",
    "Fill",
    "Populate",
    "Update",
    "Edit",
    "Revise",
    "Refactor",
    "Derive",
    "Determine",
    "Compute",
    "Calculate",
    "Find",
    "Infer",
    "Explain",
    "Describe",
    "Detail",
    "Summarize",
    "Outline",
    "Report",
    "Record",
    "Document",
    "Specify",
    "Define",
    "Set",
    "Establish",
    "Propose",
    "Suggest",
    "Author",
    "Script",
    "Craft",
    "Prepare",
    "Show",
    "Demonstrate",
    "Elaborate",
    "Illustrate",
    "Clarify",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verb sweep for first decoded token.")
    p.add_argument("--pair-dir", type=Path, default=Path("/root/data/R3/pair"))
    p.add_argument("--model-path", type=str, default="/root/data/Qwen/Qwen3-1.7B")
    p.add_argument("--output-dir", type=Path, default=Path("/root/data/R3/pair"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--verbs",
        type=str,
        default=",".join(DEFAULT_VERBS),
        help="Comma-separated verb list, e.g. 'Write,State,Provide'",
    )
    p.add_argument(
        "--strict-unpadded",
        action="store_true",
        help="Run each prompt independently (no padding) for maximum fidelity.",
    )
    return p.parse_args()


def extract_q(path: Path) -> int:
    m = re.search(r"q(\d+)", path.name)
    if not m:
        raise ValueError(f"Unable to parse q id from {path}")
    return int(m.group(1))


def load_base_prompts(pair_dir: Path) -> List[Tuple[int, str]]:
    files = sorted(pair_dir.glob("prompt-corrupted-q*.txt"), key=extract_q)
    prompts: List[Tuple[int, str]] = []
    for fp in files:
        q = extract_q(fp)
        prompts.append((q, fp.read_text()))
    if not prompts:
        raise RuntimeError(f"No prompt-corrupted-q*.txt found in {pair_dir}")
    return prompts


def replace_verb(text: str, verb: str) -> str:
    src = "State the function body in math.py based on the function definition and docstring below:"
    dst = f"{verb} the function body in math.py based on the function definition and docstring below:"
    n = text.count(src)
    if n != 1:
        raise ValueError(f"Expected exactly one target phrase, got {n}")
    return text.replace(src, dst, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    verbs = [v.strip() for v in args.verbs.split(",") if v.strip()]
    if not verbs:
        raise RuntimeError("No verbs specified.")

    base_prompts = load_base_prompts(args.pair_dir)
    num_samples = len(base_prompts)
    print(f"[info] Loaded {num_samples} samples from {args.pair_dir}")
    print(f"[info] Evaluating {len(verbs)} verbs")

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device_map = "cuda" if use_cuda else None
    print(f"[info] CUDA available: {use_cuda}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[info] Model device: {device}, dtype: {next(model.parameters()).dtype}")

    tool_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if len(tool_ids) != 1:
        raise RuntimeError(f"<tool_call> expected one token, got {tool_ids}")
    tool_id = int(tool_ids[0])
    print(f"[info] <tool_call> token id: {tool_id}")

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    start_all = time.time()
    with torch.inference_mode():
        for idx, verb in enumerate(verbs, start=1):
            verb_prompts = [(q, replace_verb(text, verb)) for q, text in base_prompts]
            hits = 0
            p_tool_sum = 0.0

            if args.strict_unpadded:
                for q, text in verb_prompts:
                    enc = tokenizer(text, return_tensors="pt")
                    enc = {k: v.to(device) for k, v in enc.items()}
                    logits = model(**enc).logits[:, -1, :]

                    top1_id = int(torch.argmax(logits, dim=-1).item())
                    p_tool = float(torch.softmax(logits.float(), dim=-1)[0, tool_id].item())
                    is_tool = int(top1_id == tool_id)

                    hits += is_tool
                    p_tool_sum += p_tool
                    detail_rows.append(
                        {
                            "verb": verb,
                            "q": int(q),
                            "top1_id": top1_id,
                            "top1_token": tokenizer.decode([top1_id], clean_up_tokenization_spaces=False),
                            "is_tool_call_top1": is_tool,
                            "p_tool_call": p_tool,
                        }
                    )
            else:
                for s in range(0, num_samples, args.batch_size):
                    chunk = verb_prompts[s : s + args.batch_size]
                    qs = [q for q, _ in chunk]
                    texts = [t for _, t in chunk]

                    enc = tokenizer(texts, return_tensors="pt", padding=True)
                    enc = {k: v.to(device) for k, v in enc.items()}

                    logits = model(**enc).logits
                    last_idx = enc["attention_mask"].sum(dim=1) - 1
                    bidx = torch.arange(logits.shape[0], device=device)
                    next_logits = logits[bidx, last_idx, :]

                    top1_ids = torch.argmax(next_logits, dim=-1)
                    p_tool = torch.softmax(next_logits.float(), dim=-1)[:, tool_id]
                    is_tool = top1_ids.eq(tool_id)

                    hits += int(is_tool.sum().item())
                    p_tool_sum += float(p_tool.sum().item())

                    for b in range(len(qs)):
                        tid = int(top1_ids[b].item())
                        detail_rows.append(
                            {
                                "verb": verb,
                                "q": int(qs[b]),
                                "top1_id": tid,
                                "top1_token": tokenizer.decode([tid], clean_up_tokenization_spaces=False),
                                "is_tool_call_top1": int(is_tool[b].item()),
                                "p_tool_call": float(p_tool[b].item()),
                            }
                        )

            green_count = hits
            green_rate = hits / num_samples
            mean_p_tool = p_tool_sum / num_samples
            summary_rows.append(
                {
                    "verb": verb,
                    "num_samples": num_samples,
                    "green_count": green_count,
                    "green_rate": green_rate,
                    "mean_p_tool": mean_p_tool,
                }
            )
            print(
                f"[{idx:02d}/{len(verbs):02d}] verb={verb:<12} "
                f"green={green_count:3d}/{num_samples} ({green_rate:.2%}) "
                f"mean_p_tool={mean_p_tool:.4f}"
            )

    summary_rows.sort(key=lambda r: (r["green_rate"], r["mean_p_tool"]), reverse=True)
    elapsed = time.time() - start_all
    print(f"[done] Total elapsed: {elapsed:.1f}s")

    summary_path = args.output_dir / "verb_sweep_qwen3_1.7b_summary.csv"
    detail_path = args.output_dir / "verb_sweep_qwen3_1.7b_detail.csv"
    config_path = args.output_dir / "verb_sweep_qwen3_1.7b_config.json"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["verb", "num_samples", "green_count", "green_rate", "mean_p_tool"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["verb", "q", "top1_id", "top1_token", "is_tool_call_top1", "p_tool_call"],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    config = {
        "model_path": args.model_path,
        "pair_dir": str(args.pair_dir),
        "output_dir": str(args.output_dir),
        "batch_size": args.batch_size,
        "strict_unpadded": args.strict_unpadded,
        "verbs": verbs,
        "num_samples": num_samples,
        "tool_token_id": tool_id,
        "tool_token_text": "<tool_call>",
        "elapsed_seconds": elapsed,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] Summary: {summary_path}")
    print(f"[done] Detail : {detail_path}")
    print(f"[done] Config : {config_path}")


if __name__ == "__main__":
    main()
