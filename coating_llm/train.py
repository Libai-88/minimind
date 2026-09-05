# -*- coding: utf-8 -*-
"""CoatingLLM 训练：领域预训练 + 专家推理 SFT（CPU）。"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from model import CoatingLLM

HERE = os.path.dirname(os.path.abspath(__file__))
from tokenizers import Tokenizer  # noqa: E402

_tok = Tokenizer.from_file(os.path.join(HERE, "model", "tokenizer.json"))
PAD, BOS, EOS = 0, 1, 2
SYSTEM_PROMPT = ("你是涂釉，涂料领域的配方性能小专家。你熟悉环氧酚醛、环氧配比方案、聚酯金黄"
                 "三个体系的配方与实测性能，掌握原料TDS/SDS与固化机理，能通过配方组成和化学知识，"
                 "结合历史数据推断新配方的性能。")


def encode(s):
    return _tok.encode(s).ids


class PretrainDataset(Dataset):
    def __init__(self, path, max_len=512):
        self.rows = []
        for line in open(path, encoding="utf-8"):
            t = json.loads(line).get("text", "").strip()
            if len(t) > 30:
                ids = encode(t)[: max_len - 2]
                self.rows.append([BOS] + ids + [EOS])
        print(f"  预训练数据 {len(self.rows)} 条")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate_pre(batch, max_len=512):
    ml = min(max(len(b) for b in batch), max_len)
    x = torch.full((len(batch), ml), PAD, dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : len(b)] = torch.tensor(b[:ml])
    return x, x.clone()


class SFTDataset(Dataset):
    def __init__(self, path, max_len=1024, system_prompt=SYSTEM_PROMPT):
        self.ml = max_len
        self.sp = system_prompt
        self.rows = []
        raw = []
        for line in open(path, encoding="utf-8"):
            conv = json.loads(line)["conversations"]
            prompt, answer = self._build(conv)
            raw.append((prompt, answer))
        raw.sort(key=lambda p: len(p[0]) + len(p[1]))
        for prompt, answer in raw:
            p_ids = encode(prompt)
            a_ids = encode(answer)
            self.rows.append((p_ids, a_ids))
        print(f"  SFT 数据 {len(self.rows)} 条")

    def _build(self, conv):
        parts = [f"<|im_start|>system\n{self.sp}<|im_end|>\n"]
        answer_parts = []
        for m in conv:
            if m["role"] == "assistant":
                parts.append("<|im_start|>assistant\n")
                answer_parts.append(f"{m['content']}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>user\n{m['content']}<|im_end|>\n")
        return "".join(parts), "".join(answer_parts)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate_sft(batch, max_len=1024):
    xs, ys = [], []
    for p_ids, a_ids in batch:
        ids = ([BOS] + p_ids + a_ids)[:max_len]
        n_p = min(len(p_ids) + 1, max_len)
        y = [-100] * min(n_p, len(ids)) + ids[n_p:]
        xs.append(ids)
        ys.append(y if y else [-100])
    ml = max(len(x) for x in xs)
    x = torch.full((len(xs), ml), PAD, dtype=torch.long)
    y = torch.full((len(ys), ml), -100, dtype=torch.long)
    for i, (xx, yy) in enumerate(zip(xs, ys)):
        x[i, : len(xx)] = torch.tensor(xx)
        y[i, : len(yy)] = torch.tensor(yy)
    return x, y


def cosine_lr(step, total, base, warm=30):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(total - warm, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def run_phase(model, ds, collate, epochs, lr, bs, out_dir, tag, best_path, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps_total = (len(ds) + bs - 1) // bs * epochs
    step = 0
    best = float("inf")
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(1, epochs + 1):
        idx = list(range(len(ds)))
        random.shuffle(idx)
        tot, nb = 0.0, 0
        t0 = time.time()
        for i in range(0, len(idx), bs):
            batch = [ds[j] for j in idx[i:i + bs]]
            x, y = collate(batch)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, steps_total, lr)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
            step += 1
            if nb % 20 == 0:
                print(f"  [{tag}] {ep}/{epochs} step {nb}, loss {loss.item():.4f}, "
                      f"lr {g['lr']:.2e}", flush=True)
        avg = tot / max(nb, 1)
        print(f"[{ep}/{epochs}] 平均 loss: {avg:.4f} ({time.time() - t0:.0f}s)", flush=True)
        ck = os.path.join(out_dir, f"{tag}_e{ep}.pt")
        torch.save(model.state_dict(), ck)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), best_path)
    print(f"{tag} 完成，best loss {best:.4f} -> {best_path}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pretrain", "sft", "all"], default="all")
    ap.add_argument("--pretrain_epochs", type=int, default=3)
    ap.add_argument("--sft_epochs", type=int, default=25)
    ap.add_argument("--sft_lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--sft_data", type=str, default=os.path.join(HERE, "dataset", "sft_coating.jsonl"))
    args = ap.parse_args()

    out = os.path.join(HERE, "out")
    best = os.path.join(out, "sft", "coating_best.pt")
    if args.phase in ("pretrain", "all"):
        model = CoatingLLM(vocab_size=10000, hidden=384, layers=6, heads=8, kv_heads=4,
                           intermediate=1024, max_seq=1024, dropout=0.1)
        ds = PretrainDataset(os.path.join(HERE, "dataset", "pretrain_coating.jsonl"), 512)
        run_phase(model, ds, collate_pre, args.pretrain_epochs, 3e-4, args.batch_size,
                  os.path.join(out, "pretrain"), "pretrain",
                  os.path.join(out, "pretrain", "coating_pre_best.pt"))
        init = os.path.join(out, "pretrain", "coating_pre_best.pt")
    else:
        init = args.init or os.path.join(out, "pretrain", "coating_pre_best.pt")
    if args.phase in ("sft", "all"):
        model = CoatingLLM(vocab_size=10000, hidden=384, layers=6, heads=8, kv_heads=4,
                           intermediate=1024, max_seq=1024, dropout=0.0)
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=True))
        ds = SFTDataset(args.sft_data, 1024)
        run_phase(model, ds, collate_sft, args.sft_epochs, args.sft_lr, args.batch_size,
                  os.path.join(out, "sft"), "sft", best)


if __name__ == "__main__":
    main()
