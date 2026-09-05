# -*- coding: utf-8 -*-
"""领域 BPE 分词器训练：3NEW TDS/SDS + 材料档案 + 配方文本 + 对话模板。"""
import glob
import json
import os
import pickle
import sys

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.pre_tokenizers import ByteLevel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TDS_DIR = os.path.join(ROOT, "3NEW", "generalization", "TDS-SDS")
PKL = os.path.join(ROOT, "3NEW", "generalization", "data", "merged_data.pkl")
OUT = os.path.join(HERE, "model", "tokenizer.json")

CHAT_SEEDS = [
    "你好！我是涂釉，涂料领域的配方性能小专家。",
    "你能做什么？我可以分析配方组成、解读原料TDS/SDS、推断新配方的性能表现。",
    "什么是T弯？T弯是评价涂膜柔韧性的指标，数值越小柔韧性越好。",
    "环氧树脂和酚醛树脂固化后会形成致密的交联网络。",
    "固化剂比例越高，交联密度越大，耐溶剂性越好，但涂膜会变脆。",
    "烘烤温度和时间决定固化转化率，工艺不足会残留低交联度。",
    "溶剂负责溶解树脂、调节粘度，挥发速率影响流平和成膜。",
    "水煮测试考察涂膜耐沸水性能，等级越高耐水性越好。",
]


def domain_texts():
    texts = []
    for p in glob.glob(os.path.join(TDS_DIR, "**", "*.md"), recursive=True):
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
            if len(t) > 50:
                texts.append(t[:2000])
        except OSError:
            pass
    with open(PKL, "rb") as f:
        d = pickle.load(f)
    for name, m in d["full_mat"].items():
        texts.append(
            f"{name}是{m.get('role','原料')}，类型{m.get('rtype','')}，固含{m.get('NV')}%，"
            f"密度{m.get('density')}，分子量{m.get('Mw')}，羟值{m.get('OHV')}，"
            f"酸值{m.get('AV')}，Tg {m.get('Tg')}°C。")
    for s in d["all_samples"]:
        comp = "、".join(f"{k}{v}%" for k, v in s["组分"].items())
        texts.append(f"{s['体系']}体系样本{s['样本ID']}配方：{comp}，"
                     f"烘烤{s.get('烘烤温度')}°C×{s.get('烘烤时间')}min。")
    texts.extend(CHAT_SEEDS)
    return texts


def main():
    corpus = domain_texts()
    print(f"语料: {len(corpus)} 段, 共 {sum(len(t) for t in corpus)} 字符")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=10000 - 3,
        special_tokens=["<|pad|>", "<|im_start|>", "<|im_end|>"],
        show_progress=False,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok.save(OUT)
    for probe in ["<|im_start|>user\n什么是T弯？<|im_end|>",
                  "环氧酚醛样本R01-01：IR190 66%、RF516 2.61%，烘烤200°C×10min，MEK擦拭2次、水煮4级",
                  "交联密度决定耐溶剂性：酚醛比例越高，MEK擦拭次数越多，T弯数值越大（变脆）。",
                  "0.85 1.4e-3 550 205"]:
        ids = tok.encode(probe).ids
        rt = tok.decode(ids, skip_special_tokens=False)
        assert rt == probe, f"往返不一致: {rt!r}"
        print(f"[OK] {len(probe)}字 -> {len(ids)} tokens: {probe[:40]}")
    multi_digit = [t for t, i in tok.get_vocab().items() if t.isdigit() and len(t) > 1]
    assert not multi_digit, f"存在多位数字合并token: {multi_digit[:8]}"
    print("数字均为单token: OK")
    print(f"词表 {tok.get_vocab_size()}，已保存 {OUT}")


if __name__ == "__main__":
    main()
