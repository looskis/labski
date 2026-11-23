#!/usr/bin/env python3
"""Train and evaluate Inkling-2-style recurrent loop retrofits on open-weight LLMs.

Runs on a single NVIDIA DGX Spark (GB10, 121 GiB unified memory) inside
vllm/vllm-openai:v0.33.0 (PyTorch 2.14 + transformers 5.20.2).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import subprocess
import threading
import time
from typing import Any

import torch

from inkling_loop import (
    InklingLoopConfig,
    InklingLoopedCausalLM,
    default_loop_layout,
    load_text_causal_lm,
)
from tasks import (
    TaskExample,
    build_serial_eval_suite,
    build_serial_train_examples,
    extract_gsm8k_gold,
    format_gsm8k_direct,
    load_jsonl,
    normalize_answer_str,
)


# ---------------------------------------------------------------------------
# GPU Power / Energy Sampler (nvidia-smi background poller)
# ---------------------------------------------------------------------------

class GpuPowerMonitor:
    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._thread: threading.Thread | None = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                ).strip()
                val = float(out.splitlines()[0].strip())
                self._samples.append((time.perf_counter(), val))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict[str, float | int]:
        if len(self._samples) < 2:
            return {"samples": len(self._samples), "mean_watts": 0.0, "energy_joules": 0.0}
        energy = 0.0
        for (t0, w0), (t1, w1) in zip(self._samples[:-1], self._samples[1:]):
            energy += 0.5 * (w0 + w1) * (t1 - t0)
        mean_w = sum(w for _, w in self._samples) / len(self._samples)
        return {
            "samples": len(self._samples),
            "mean_watts": round(mean_w, 2),
            "energy_joules": round(energy, 2),
        }


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def encode_with_bos(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    return ids


def encode_supervised(tokenizer, prompt: str, answer: str, add_newline: bool = True) -> tuple[list[int], list[int]]:
    """Encode (prompt, answer) into (input_ids, labels) with prompt tokens masked to -100."""
    target = answer + ("\n" if add_newline else "")
    prompt_ids = encode_with_bos(tokenizer, prompt)
    full_ids = encode_with_bos(tokenizer, prompt + target)
    # Guard against BPE boundary merge shifting prompt length
    ans_start = len(prompt_ids)
    while ans_start > 0 and (ans_start >= len(full_ids) or full_ids[:ans_start] != prompt_ids[:ans_start]):
        ans_start -= 1
    if ans_start >= len(full_ids):
        ans_ids = tokenizer.encode(target, add_special_tokens=False)
        full_ids = prompt_ids + ans_ids
        ans_start = len(prompt_ids)
    labels = [-100] * ans_start + full_ids[ans_start:]
    return full_ids, labels


def encode_lm_passage(tokenizer, text: str, max_tokens: int = 256) -> tuple[list[int], list[int]]:
    ids = encode_with_bos(tokenizer, text)[:max_tokens]
    return ids, list(ids)


# ---------------------------------------------------------------------------
# Evaluation Routines
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_serial_suite(
    model: InklingLoopedCausalLM,
    tokenizer,
    examples: list[TaskExample],
    num_passes: int,
    message_mode: str = "normal",
    max_new_tokens: int = 4,
    collect_first_diag: bool = False,
) -> dict[str, Any]:
    """Evaluate exact-match greedy accuracy AND exact gold-answer NLL on the serial suite."""
    model.eval()
    device = next(model.parameters()).device
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))
    stop_ids = {tokenizer.eos_token_id} | newline_ids

    by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    sample_diag = None

    for idx, ex in enumerate(examples):
        # 1. Single causal forward on (prompt + gold answer) gives BOTH exact gold-answer NLL
        #    AND the exact 1st greedy token at position (prompt_len - 1)!
        prompt_ids = encode_with_bos(tokenizer, ex.prompt)
        full_ids, labels = encode_supervised(tokenizer, ex.prompt, ex.answer, add_newline=False)
        inp_full = torch.tensor([full_ids], dtype=torch.long, device=device)
        lbl_full = torch.tensor([labels], dtype=torch.long, device=device)
        want_diag = collect_first_diag and (idx == 0)
        # Keep logits for the answer positions + position right before answer (prompt_len - 1)
        prompt_len = len(prompt_ids)
        can_reuse_tok1 = (len(full_ids) > prompt_len) and (full_ids[:prompt_len] == prompt_ids)
        if can_reuse_tok1:
            keep = len(full_ids) - prompt_len + 1
            out_fwd = model(
                input_ids=inp_full,
                num_passes=num_passes,
                message_mode=message_mode,
                logits_to_keep=keep,
                collect_diagnostics=want_diag,
            )
            ans_logits = out_fwd["logits"][0]  # shape (keep, V); index 0 predicts full_ids[prompt_len]
            ans_targets = inp_full[0, prompt_len:]
            ans_nll = float(torch.nn.functional.cross_entropy(ans_logits[:-1], ans_targets).item())
            first_tok = ans_logits[0].argmax(dim=-1).view(1, 1)
            if want_diag:
                sample_diag = out_fwd["diagnostics"]

            tok_id = int(first_tok.item())
            piece = tokenizer.decode([tok_id], skip_special_tokens=False)
            if tok_id in stop_ids or "\n" in piece or "\r" in piece or "|" in piece or max_new_tokens <= 1:
                gen_ids = first_tok
            else:
                inp_p = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                seq_with_1 = torch.cat([inp_p, first_tok], dim=-1)
                rest = model.greedy_generate(
                    seq_with_1,
                    max_new_tokens=max_new_tokens - 1,
                    num_passes=num_passes,
                    message_mode=message_mode,
                    stop_token_ids=stop_ids,
                    tokenizer=tokenizer,
                )
                gen_ids = torch.cat([first_tok, rest], dim=-1)
        else:
            out_nll = model(
                input_ids=inp_full,
                labels=lbl_full,
                num_passes=num_passes,
                message_mode=message_mode,
                collect_diagnostics=want_diag,
            )
            ans_nll = float(out_nll["loss"].item())
            if want_diag:
                sample_diag = out_nll["diagnostics"]
            inp_p = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            gen_ids = model.greedy_generate(
                inp_p,
                max_new_tokens=max_new_tokens,
                num_passes=num_passes,
                message_mode=message_mode,
                stop_token_ids=stop_ids,
                tokenizer=tokenizer,
            )
        raw_pred = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
        pred_norm = normalize_answer_str(raw_pred)
        gold_norm = ex.meta["target"]
        correct = int(pred_norm == gold_norm)

        row = {
            "task": ex.task,
            "split": ex.split,
            "depth": ex.depth,
            "gold": gold_norm,
            "pred": pred_norm,
            "raw_pred": raw_pred[:32],
            "correct": correct,
            "ans_nll": round(ans_nll, 5),
        }
        all_rows.append(row)
        by_key[(ex.task, ex.split, ex.depth)].append(row)

    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / max(n, 1)
        nll = sum(r["ans_nll"] for r in rows) / max(n, 1)
        return {"n": n, "correct": sum(r["correct"] for r in rows), "acc": round(acc, 4), "mean_ans_nll": round(nll, 4)}

    id_rows = [r for r in all_rows if r["split"] == "eval_id"]
    ood_rows = [r for r in all_rows if r["split"] == "eval_ood"]
    per_task = {}
    for task in sorted({r["task"] for r in all_rows}):
        t_id = [r for r in id_rows if r["task"] == task]
        t_ood = [r for r in ood_rows if r["task"] == task]
        per_depth = {}
        for (t, sp, d), rows in sorted(by_key.items()):
            if t == task:
                per_depth[f"{sp}_d{d}"] = _agg(rows)
        per_task[task] = {
            "id": _agg(t_id),
            "ood": _agg(t_ood),
            "by_depth": per_depth,
        }

    return {
        "num_passes": num_passes,
        "message_mode": message_mode,
        "overall": _agg(all_rows),
        "overall_id": _agg(id_rows),
        "overall_ood": _agg(ood_rows),
        "per_task": per_task,
        "diagnostics": sample_diag,
        "rows": all_rows,
    }


@torch.no_grad()
def evaluate_wikitext(
    model: InklingLoopedCausalLM,
    tokenizer,
    passages: list[dict[str, Any]],
    num_passes: int,
    max_tokens: int = 256,
    limit: int = 24,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_toks = 0
    per_passage = []
    for row in passages[:limit]:
        ids, labels = encode_lm_passage(tokenizer, row["text"], max_tokens=max_tokens)
        if len(ids) < 8:
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        lbl = torch.tensor([labels], dtype=torch.long, device=device)
        out = model(input_ids=inp, labels=lbl, num_passes=num_passes)
        n_scored = len(ids) - 1
        loss_val = float(out["loss"].item())
        total_nll += loss_val * n_scored
        total_toks += n_scored
        per_passage.append(round(loss_val, 5))
    mean_nll = total_nll / max(total_toks, 1)
    return {
        "num_passes": num_passes,
        "passages": len(per_passage),
        "scored_tokens": total_toks,
        "mean_nll": round(mean_nll, 5),
        "perplexity": round(math.exp(min(mean_nll, 20.0)), 4),
        "per_passage_nll": per_passage,
    }


@torch.no_grad()
def evaluate_gsm8k_direct(
    model: InklingLoopedCausalLM,
    tokenizer,
    rows: list[dict[str, Any]],
    num_passes: int,
    limit: int = 48,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))
    stop_ids = {tokenizer.eos_token_id} | newline_ids
    correct = 0
    total = 0
    nlls = []
    for row in rows[:limit]:
        prompt, ans = format_gsm8k_direct(row)
        gold = extract_gsm8k_gold(row["answer"])
        full_ids, labels = encode_supervised(tokenizer, prompt, ans, add_newline=False)
        inp_f = torch.tensor([full_ids], dtype=torch.long, device=device)
        lbl_f = torch.tensor([labels], dtype=torch.long, device=device)
        nll = float(model(input_ids=inp_f, labels=lbl_f, num_passes=num_passes)["loss"].item())
        nlls.append(nll)

        p_ids = encode_with_bos(tokenizer, prompt)
        inp_p = torch.tensor([p_ids], dtype=torch.long, device=device)
        gen = model.greedy_generate(
            inp_p, max_new_tokens=6, num_passes=num_passes, stop_token_ids=stop_ids, tokenizer=tokenizer
        )
        pred = normalize_answer_str(tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)).replace(",", "")
        if pred == gold:
            correct += 1
        total += 1
    return {
        "num_passes": num_passes,
        "n": total,
        "correct": correct,
        "acc": round(correct / max(total, 1), 4),
        "mean_ans_nll": round(sum(nlls) / max(len(nlls), 1), 4),
    }


@torch.no_grad()
def benchmark_forward_cost(
    model: InklingLoopedCausalLM,
    tokenizer,
    sample_prompt: str,
    pass_counts: tuple[int, ...] = (1, 2, 3),
    repeats: int = 8,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    ids = encode_with_bos(tokenizer, sample_prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    results = {}
    for p in pass_counts:
        # Warmup
        for _ in range(2):
            _ = model(input_ids=inp, num_passes=p, logits_to_keep=1)
        torch.cuda.synchronize()
        with GpuPowerMonitor(interval_s=0.1) as pm:
            t0 = time.perf_counter()
            for _ in range(repeats):
                _ = model(input_ids=inp, num_passes=p, logits_to_keep=1)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        pstats = pm.summary()
        ms_per_fwd = (dt / repeats) * 1000.0
        j_per_fwd = pstats["energy_joules"] / repeats if pstats["samples"] >= 2 else 0.0
        results[f"P{p}"] = {
            "seq_len": len(ids),
            "ms_per_forward": round(ms_per_fwd, 2),
            "toks_per_sec": round((len(ids) * repeats) / dt, 1),
            "mean_watts": pstats["mean_watts"],
            "joules_per_forward": round(j_per_fwd, 3),
        }
    return results


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_loop_retrofit(
    model: InklingLoopedCausalLM,
    tokenizer,
    serial_train: list[TaskExample],
    wt_train: list[dict[str, Any]],
    gsm_train: list[dict[str, Any]],
    serial_val: list[TaskExample],
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    model.gradient_checkpointing = True

    # Separate param groups: message_reader.query benefits from 3x LR due to 1/sqrt(H) scaling
    reader_params = list(model.message_reader.parameters()) if model.message_reader is not None else []
    reader_ids = {id(p) for p in reader_params}
    other_params = [p for p in model.trainable_parameters() if id(p) not in reader_ids]

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if reader_params:
        param_groups.append({"params": reader_params, "lr": args.lr * 3.0, "weight_decay": 0.0})

    if not param_groups:
        print("No trainable parameters (pure zero-shot / naive_loop eval).", flush=True)
        return []

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(args.warmup_steps, 1))
        progress = float(step - args.warmup_steps) / float(max(args.steps - args.warmup_steps, 1))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    rng = random.Random(args.seed)
    serial_idx = 0
    wt_idx = 0
    gsm_idx = 0
    train_log: list[dict[str, Any]] = []
    log_path = out_dir / "train_log.jsonl"

    t_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_serial_loss = 0.0
        step_lm_loss = 0.0
        n_serial_mb = 0
        n_lm_mb = 0

        for mb in range(args.grad_accum):
            # Decide pass count for this microbatch
            if args.mode == "lora_only":
                p_cur = 1
            elif args.varloop:
                if model.has_core_lora:
                    p_cur = rng.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0]
                else:
                    p_cur = rng.choices([2, 3], weights=[0.35, 0.65])[0]
            else:
                p_cur = args.num_passes

            # 75% serial multi-hop tasks, 25% real LM/GSM8K regularizer
            is_lm = (mb % 4 == 3) and (wt_train or gsm_train)
            if is_lm:
                if gsm_train and (mb // 4 + step) % 2 == 0:
                    row = gsm_train[gsm_idx % len(gsm_train)]
                    gsm_idx += 1
                    prompt, ans = format_gsm8k_direct(row)
                    ids, labels = encode_supervised(tokenizer, prompt, ans, add_newline=True)
                else:
                    row = wt_train[wt_idx % len(wt_train)]
                    wt_idx += 1
                    ids, labels = encode_lm_passage(tokenizer, row["text"], max_tokens=224)
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                lbl = torch.tensor([labels], dtype=torch.long, device=device)
                loss = model(input_ids=inp, labels=lbl, num_passes=p_cur)["loss"]
                scaled = (loss * args.lm_weight) / args.grad_accum
                if scaled.requires_grad:
                    scaled.backward()
                step_lm_loss += float(loss.detach().item())
                n_lm_mb += 1
            else:
                ex = serial_train[serial_idx % len(serial_train)]
                serial_idx += 1
                if serial_idx % len(serial_train) == 0:
                    rng.shuffle(serial_train)
                ids, labels = encode_supervised(tokenizer, ex.prompt, ex.answer, add_newline=True)
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                lbl = torch.tensor([labels], dtype=torch.long, device=device)
                loss = model(input_ids=inp, labels=lbl, num_passes=p_cur)["loss"]
                scaled = loss / args.grad_accum
                if scaled.requires_grad:
                    scaled.backward()
                step_serial_loss += float(loss.detach().item())
                n_serial_mb += 1

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0).item())
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - t_start
            rec: dict[str, Any] = {
                "step": step,
                "elapsed_s": round(elapsed, 1),
                "sec_per_step": round(elapsed / step, 2),
                "lr": round(scheduler.get_last_lr()[0], 6),
                "serial_loss": round(step_serial_loss / max(n_serial_mb, 1), 4),
                "lm_loss": round(step_lm_loss / max(n_lm_mb, 1), 4) if n_lm_mb else None,
                "grad_norm": round(grad_norm, 4),
                "peak_mem_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
            }
            if step % args.eval_every == 0 and step < args.steps and serial_val:
                val_p = 1 if args.mode == "lora_only" else args.num_passes
                val_res = evaluate_serial_suite(model, tokenizer, serial_val, num_passes=val_p)
                rec["val_id_acc"] = val_res["overall_id"]["acc"]
                rec["val_ood_acc"] = val_res["overall_ood"]["acc"]
                rec["val_id_nll"] = val_res["overall_id"]["mean_ans_nll"]
                rec["val_ood_nll"] = val_res["overall_ood"]["mean_ans_nll"]
                model.save_adapter(out_dir / f"ckpt-step{step}")

            train_log.append(rec)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)

    return train_log


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local snapshot directory or HF model ID")
    p.add_argument("--model-label", required=True, help="Human-readable label for results")
    p.add_argument("--run-name", required=True, help="Subdirectory under --out-root")
    p.add_argument("--out-root", type=Path, default=Path("/var/tmp/inkling-loop-exp/runs"))
    p.add_argument("--data-dir", type=Path, default=Path("/var/tmp/inkling-loop-exp/data"))
    p.add_argument("--mode", choices=("inkling2", "naive_loop", "scalar_gate_loop", "lora_only"), default="inkling2")
    p.add_argument("--num-passes", type=int, default=3)
    p.add_argument("--varloop", action="store_true", help="Sample pass counts P in {2,3} during training")
    p.add_argument("--gate-rank", type=int, default=64)
    p.add_argument("--core-lora-rank", type=int, default=0)
    p.add_argument("--loop-start", type=int, default=None)
    p.add_argument("--loop-end", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--lm-weight", type=float, default=0.25)
    p.add_argument("--train-per-task", type=int, default=600)
    p.add_argument("--eval-per-depth", type=int, default=16)
    p.add_argument("--val-subset", type=int, default=48)
    p.add_argument("--wt-eval-limit", type=int, default=24)
    p.add_argument("--gsm-eval-limit", type=int, default=48)
    p.add_argument("--eval-passes", default="1,2,3,4,5", help="Comma-separated pass counts to evaluate")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-ablation", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = args.out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Starting run: {args.run_name} ({args.model_label}) mode={args.mode} ===", flush=True)

    t0 = time.perf_counter()
    base_model, tokenizer, family = load_text_causal_lm(args.model, dtype=torch.bfloat16, device="cuda")
    load_s = time.perf_counter() - t0
    tc = base_model.model.config
    n_layers = tc.num_hidden_layers
    hidden_size = tc.hidden_size
    d_start, d_end, d_blk = default_loop_layout(family, n_layers)
    loop_start = d_start if args.loop_start is None else args.loop_start
    loop_end = d_end if args.loop_end is None else args.loop_end
    block_size = d_blk if args.block_size is None else args.block_size

    loop_cfg = InklingLoopConfig(
        backbone_family=family,
        model_id=args.model_label,
        num_hidden_layers=n_layers,
        hidden_size=hidden_size,
        loop_start=loop_start,
        loop_end=loop_end,
        block_size=block_size,
        num_passes=args.num_passes,
        gate_rank=args.gate_rank,
        rms_norm_eps=getattr(tc, "rms_norm_eps", 1e-6),
        mode=args.mode,
        core_lora_rank=args.core_lora_rank,
    )
    model = InklingLoopedCausalLM(base_model, loop_cfg)
    param_stats = model.count_parameters()
    print(f"Loaded {args.model_label} ({family}, L={n_layers}, H={hidden_size}) in {load_s:.1f}s", flush=True)
    print(f"Loop layout: prelude [0..{loop_start-1}], core [{loop_start}..{loop_end-1}] (block={block_size}), coda [{loop_end}..{n_layers-1}]", flush=True)
    print(f"Params: {json.dumps(param_stats)}", flush=True)

    # Datasets
    serial_train = build_serial_train_examples(seed=1001, per_task=args.train_per_task)
    serial_eval = build_serial_eval_suite(seed=20261122, per_depth=args.eval_per_depth)
    # Deterministic small validation slice (balanced across tasks & ID/OOD)
    val_rng = random.Random(777)
    val_id = [e for e in serial_eval if e.split == "eval_id"]
    val_ood = [e for e in serial_eval if e.split == "eval_ood"]
    serial_val = val_rng.sample(val_id, min(args.val_subset // 2, len(val_id))) + val_rng.sample(val_ood, min(args.val_subset // 2, len(val_ood)))

    wt_train = load_jsonl(args.data_dir / "prose_train.jsonl") or load_jsonl(args.data_dir / "wikitext2_train.jsonl")
    wt_test = load_jsonl(args.data_dir / "prose_test.jsonl") or load_jsonl(args.data_dir / "wikitext2_test.jsonl")
    gsm_train = load_jsonl(args.data_dir / "gsm8k_train.jsonl")
    gsm_test = load_jsonl(args.data_dir / "gsm8k_test.jsonl")
    print(f"Data: serial_train={len(serial_train)}, serial_eval={len(serial_eval)}, wt_train={len(wt_train)}, wt_test={len(wt_test)}, gsm_train={len(gsm_train)}, gsm_test={len(gsm_test)}", flush=True)

    # 1. Step-0 mathematical identity verification & residual norm profile
    sample_ids = torch.tensor([encode_with_bos(tokenizer, serial_eval[0].prompt)], dtype=torch.long, device="cuda")
    step0_check = model.verify_step0_identity(sample_ids)
    rms_profile = model.measure_layer_rms_profile(sample_ids)
    print(f"Step-0 identity check: {json.dumps(step0_check)}", flush=True)

    # 2. Hardware forward latency & GPU energy benchmark across P=1,2,3
    hw_passes = (1,) if args.mode == "lora_only" else (1, 2, 3)
    hw_bench = benchmark_forward_cost(model, tokenizer, serial_eval[0].prompt, pass_counts=hw_passes, repeats=6)
    print(f"Hardware forward cost: {json.dumps(hw_bench)}", flush=True)

    # 3. Pre-training Baseline (P=1 exact pretrained model; and P=3 for naive_loop to show step-0 shock!)
    print("Evaluating pre-training P=1 baseline...", flush=True)
    baseline_serial_p1 = evaluate_serial_suite(model, tokenizer, serial_eval, num_passes=1)
    baseline_wt_p1 = evaluate_wikitext(model, tokenizer, wt_test, num_passes=1, limit=args.wt_eval_limit) if wt_test else None
    baseline_gsm_p1 = evaluate_gsm8k_direct(model, tokenizer, gsm_test, num_passes=1, limit=args.gsm_eval_limit) if gsm_test else None
    print(
        f"Baseline P=1 | ID acc={baseline_serial_p1['overall_id']['acc']} (nll={baseline_serial_p1['overall_id']['mean_ans_nll']}) "
        f"| OOD acc={baseline_serial_p1['overall_ood']['acc']} (nll={baseline_serial_p1['overall_ood']['mean_ans_nll']}) "
        f"| WT ppl={baseline_wt_p1['perplexity'] if baseline_wt_p1 else 'N/A'} "
        f"| GSM8K direct={baseline_gsm_p1['acc'] if baseline_gsm_p1 else 'N/A'}",
        flush=True,
    )

    baseline_naive_p3 = None
    if args.mode == "naive_loop":
        print("Evaluating pre-training naive_loop P=3 (shows step-0 representation shock)...", flush=True)
        baseline_naive_p3 = evaluate_serial_suite(model, tokenizer, serial_eval, num_passes=3)
        print(f"Naive-loop step-0 P=3 | ID acc={baseline_naive_p3['overall_id']['acc']} | OOD acc={baseline_naive_p3['overall_ood']['acc']}", flush=True)

    # 4. Train!
    torch.cuda.reset_peak_memory_stats()
    train_t0 = time.perf_counter()
    train_log = train_loop_retrofit(
        model=model,
        tokenizer=tokenizer,
        serial_train=serial_train,
        wt_train=wt_train,
        gsm_train=gsm_train,
        serial_val=serial_val,
        args=args,
        out_dir=out_dir,
    )
    train_elapsed_s = round(time.perf_counter() - train_t0, 1)
    peak_train_mem_gib = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    model.save_adapter(out_dir / "final_adapter")

    # 5. Post-training multi-pass depth sweep evaluation
    eval_passes = [1] if args.mode == "lora_only" else [int(x) for x in args.eval_passes.split(",") if x.strip()]
    post_serial: dict[str, Any] = {}
    post_wt: dict[str, Any] = {}
    post_gsm: dict[str, Any] = {}
    all_pred_rows: dict[str, list[dict[str, Any]]] = {}

    for p_eval in eval_passes:
        if p_eval == 1 and not model.has_core_lora:
            # Base weights are frozen and P=1 executes no loop gates -> mathematically identical to baseline_p1
            with torch.no_grad():
                check_diff = float((
                    model(input_ids=sample_ids, num_passes=1, logits_to_keep=1)["logits"]
                    - model.base_model(input_ids=sample_ids, use_cache=False).logits[:, -1:, :].float()
                ).abs().max().item())
            print(f"Post-training P=1 verified frozen (max diff vs stock={check_diff:.5f}); reusing exact baseline_p1.", flush=True)
            post_serial["P1"] = {k: v for k, v in baseline_serial_p1.items() if k != "rows"}
            all_pred_rows["P1"] = list(baseline_serial_p1["rows"])
            if wt_test:
                post_wt["P1"] = baseline_wt_p1
            if gsm_test:
                post_gsm["P1"] = baseline_gsm_p1
            continue
        print(f"Post-training eval at P={p_eval}...", flush=True)
        res = evaluate_serial_suite(
            model,
            tokenizer,
            serial_eval,
            num_passes=p_eval,
            message_mode="normal",
            collect_first_diag=(p_eval == args.num_passes),
        )
        all_pred_rows[f"P{p_eval}"] = res.pop("rows")
        post_serial[f"P{p_eval}"] = res
        print(
            f"  Post P={p_eval} | ID acc={res['overall_id']['acc']} (nll={res['overall_id']['mean_ans_nll']}) "
            f"| OOD acc={res['overall_ood']['acc']} (nll={res['overall_ood']['mean_ans_nll']})",
            flush=True,
        )
        if p_eval in (1, 2, 3):
            if wt_test:
                post_wt[f"P{p_eval}"] = evaluate_wikitext(model, tokenizer, wt_test, num_passes=p_eval, limit=args.wt_eval_limit)
            if gsm_test:
                post_gsm[f"P{p_eval}"] = evaluate_gsm8k_direct(model, tokenizer, gsm_test, num_passes=p_eval, limit=args.gsm_eval_limit)

    # 6. Causal message ablations at P=num_passes (proves genuine cross-pass information flow!)
    causal_ablations: dict[str, Any] = {}
    if not args.skip_ablation and args.mode in ("inkling2", "scalar_gate_loop"):
        for mmode in ("zero", "shuffle_tokens", "freeze_pass1"):
            print(f"Causal ablation at P={args.num_passes} with message_mode={mmode}...", flush=True)
            ares = evaluate_serial_suite(
                model, tokenizer, serial_eval, num_passes=args.num_passes, message_mode=mmode
            )
            ares.pop("rows")
            causal_ablations[mmode] = ares
            print(
                f"  Ablation ({mmode}) | ID acc={ares['overall_id']['acc']} (nll={ares['overall_id']['mean_ans_nll']}) "
                f"| OOD acc={ares['overall_ood']['acc']} (nll={ares['overall_ood']['mean_ans_nll']})",
                flush=True,
            )

    # Strip verbose rows from baseline before saving summary metrics.json
    base_p1_rows = baseline_serial_p1.pop("rows")
    if baseline_naive_p3 is not None:
        baseline_naive_p3.pop("rows")

    summary = {
        "run_name": args.run_name,
        "model_label": args.model_label,
        "model_path": args.model,
        "hostname": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0),
        "loop_config": loop_cfg.to_dict(),
        "train_args": {
            "mode": args.mode,
            "num_passes": args.num_passes,
            "varloop": args.varloop,
            "steps": args.steps,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "lm_weight": args.lm_weight,
            "seed": args.seed,
        },
        "param_stats": param_stats,
        "load_time_s": round(load_s, 1),
        "train_elapsed_s": train_elapsed_s,
        "peak_train_mem_gib": peak_train_mem_gib,
        "step0_identity_check": step0_check,
        "layer_rms_profile": rms_profile,
        "hardware_forward_cost": hw_bench,
        "baseline_p1": {
            "serial": baseline_serial_p1,
            "wikitext2": baseline_wt_p1,
            "gsm8k_direct": baseline_gsm_p1,
        },
        "baseline_naive_p3": baseline_naive_p3,
        "post_training": {
            "serial": post_serial,
            "wikitext2": post_wt,
            "gsm8k_direct": post_gsm,
        },
        "causal_ablations": causal_ablations,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "eval_predictions.jsonl").open("w") as f:
        for tag, rows in [("baseline_P1", base_p1_rows), *all_pred_rows.items()]:
            for r in rows:
                f.write(json.dumps({"eval_tag": tag, **r}) + "\n")

    print(f"=== DONE {args.run_name}! Saved to {out_dir / 'metrics.json'} ===", flush=True)


if __name__ == "__main__":
    main()
