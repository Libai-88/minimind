# -*- coding: utf-8 -*-
"""涂釉对话推理接口：交互式 / 单次提问。"""
import argparse
import os
import sys

import torch

from model import CoatingLLM
from train import SYSTEM_PROMPT, encode
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
PAD, BOS, EOS = 0, 1, 2


def clean_reply(text):
    for stop in ("<|im_end|>", "<|im_start|>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip().strip("assistant").strip()


def build_prompt(history, question, system_prompt=SYSTEM_PROMPT):
    s = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for q, a in history[-3:]:
        s += f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    s += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return s


def chat_once(model, tok, question, history=(), temperature=0.5, top_p=0.9, top_k=30,
              repetition_penalty=1.05, max_new_tokens=400, system_prompt=SYSTEM_PROMPT):
    prompt = build_prompt(history, question, system_prompt)
    ids = encode(prompt)
    ctx = torch.tensor([[BOS] + ids[-(model.max_seq - 2):]], dtype=torch.long)
    out = model.generate(ctx, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                         eos_token_id=EOS)
    reply_ids = out[0, ctx.shape[1]:].tolist()
    if EOS in reply_ids:
        reply_ids = reply_ids[: reply_ids.index(EOS)]
    return clean_reply(tok.decode(reply_ids, skip_special_tokens=False))


def find_weight():
    sft = os.path.join(HERE, "out", "sft")
    best = os.path.join(sft, "coating_best.pt")
    if os.path.exists(best):
        return best
    pts = sorted((os.path.getmtime(os.path.join(sft, f)), os.path.join(sft, f))
                 for f in os.listdir(sft) if f.endswith(".pt"))
    return pts[-1][1] if pts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(HERE, "model", "tokenizer.json"))
    model = CoatingLLM(vocab_size=10000, hidden=384, layers=6, heads=8, kv_heads=4,
                       intermediate=1024, max_seq=1024, dropout=0.0)
    w = find_weight()
    model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
    model.eval()
    print(f"[已加载] {w}")

    if args.query:
        t = args.temperature if not args.greedy else 0.01
        print("涂釉:", chat_once(model, tok, args.query, temperature=t,
                                 top_p=args.top_p, top_k=args.top_k,
                                 repetition_penalty=args.repetition_penalty,
                                 max_new_tokens=args.max_new_tokens))
        return
    history = []
    print("涂釉已上线（输入 q 退出）")
    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q in ("q", "quit", "exit"):
            break
        a = chat_once(model, tok, q, history, temperature=args.temperature,
                      top_p=args.top_p, top_k=args.top_k,
                      repetition_penalty=args.repetition_penalty,
                      max_new_tokens=args.max_new_tokens)
        history.append((q, a))
        print("涂釉:", a)


if __name__ == "__main__":
    main()
