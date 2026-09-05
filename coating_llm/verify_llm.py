# -*- coding: utf-8 -*-
"""涂釉(CoatingLLM)专项检验：思维链正确性 / 推理准确性 / 新情形举一反三。

A. 思维链正确性：对未参与"分析链"训练的真实样本提问，核对
   机理量(当量比/交联密度/Tg/固含) vs mech_readout 真值、
   历史引用样本与数值 vs 实测库、链结构五要素完整性。
B. 推理准确性：全新扰动配方(新随机种子，保证不在任何训练集)上，
   涂釉区间结论 vs GBM裁判(CV已验证) 点预测的覆盖率与区间宽度。
C. 举一反三：固化剂+50%/减半的因果方向、烘烤升级方向、
   极端配方的坦诚度(不编数值)、低资源体系(聚酯金黄)的机理量迁移。
"""
import json
import os
import random
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "coating_ai"))

from model import CoatingLLM  # noqa: E402
from chat import chat_once  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from data import mat_lib, samples_df, labeled  # noqa: E402
from features import build_matrix, build_row, mech_readout  # noqa: E402

SEED = 2026
torch.set_num_threads(3)


def load_model():
    tok = Tokenizer.from_file(os.path.join(HERE, "model", "tokenizer.json"))
    model = CoatingLLM(vocab_size=10000, hidden=256, layers=4, heads=4, kv_heads=2,
                       intermediate=640, max_seq=1024, dropout=0.0)
    w = os.path.join(HERE, "out", "sft", "coating_best.pt")
    model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
    model.eval()
    return model, tok


def ask(model, tok, q):
    return chat_once(model, tok, q, temperature=0.01, top_p=0.9, top_k=30,
                     repetition_penalty=1.05, max_new_tokens=320)


def comp_str(comp):
    return "、".join(f"{k} {v:.1f}%" for k, v in sorted(comp.items(), key=lambda x: -x[1]))


def infer_q(system, comp, bake):
    return (f"请推断这个新配方的性能：{system}体系，{comp_str(comp)}，"
            f"烘烤{bake[0]:.0f}°C×{bake[1]:.0f}min。")


# ---------------- 训练集登记：哪些样本/配方进过训练 ----------------
def trained_registry():
    ids, qs = set(), []
    for name in ("sft_coating.jsonl", "sft_refine.jsonl"):
        p = os.path.join(HERE, "dataset", name)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            u = json.loads(line)["conversations"][0]["content"]
            m = re.match(r"分析一下样本(.+?)，它的性能是怎么来的？", u)
            if m:
                ids.add(m.group(1))
            if "请推断这个新配方的性能" in u or "帮我看看这个配方" in u or "预测一下这个配方" in u:
                qs.append(u)
    return ids, qs


def trained_comp_keys(qs):
    keys = set()
    for u in qs:
        pairs = re.findall(r"([^\s、（）]+) (\d+(?:\.\d+)?)%", u)
        if pairs:
            keys.add(tuple(sorted((k, round(float(v), 1)) for k, v in pairs)))
    return keys


# ---------------- 解析涂釉的回答 ----------------
def parse_mech(text):
    out = {}
    m = re.search(r"(?:环氧当量比|NCO/OH当量比)约([\d.]+)", text)
    if m:
        out["ratio"] = float(m.group(1))
    m = re.search(r"交联密度量级([\d.]+)e-3", text)
    if m:
        out["ne"] = float(m.group(1))
    m = re.search(r"固含(\d+)%", text)
    if m:
        out["solids"] = float(m.group(1))
    m = re.search(r"Tg约(\d+)°C", text)
    if m:
        out["tg"] = float(m.group(1))
    return out


def parse_intervals(text):
    t = re.search(r"T弯约(\d+)~(\d+)mm", text)
    m = re.search(r"MEK擦拭约(\d+)~(\d+)次", text) or re.search(r"MEK约(\d+)~(\d+)次", text)
    out = {}
    if t:
        out["t"] = (float(t.group(1)), float(t.group(2)))
    if m:
        out["m"] = (float(m.group(1)), float(m.group(2)))
    return out


def chain_structure(text):
    has_comp = "占" in text
    has_mech = "当量比" in text or "交联密度" in text
    has_ref = "对照历史" in text or re.search(r"[A-Za-z0-9\-]{3,}（\d", text)
    has_concl = re.search(r"T弯|MEK|水煮", text) is not None
    has_bake = "烘烤" in text or "°C" in text
    return sum([has_comp, has_mech, bool(has_ref), has_concl, has_bake]), \
        [has_comp, has_mech, bool(has_ref), has_concl, has_bake]


# ---------------- 测试A：思维链正确性 ----------------
def test_a(model, tok, df, mlib, trained_ids):
    cand = df[~df["样本ID"].isin(trained_ids)]
    cand = cand.iloc[:12]
    n_ratio = n_ne = n_sol = n_tg = 0
    t_ratio = t_ne = t_sol = t_tg = 0
    cite_ok = cite_all = 0
    struct = []
    for _, r in cand.iterrows():
        comp = {k: float(v) for k, v in r["组分"].items()}
        mech = mech_readout(comp, mlib, r["烘烤温度"], r["烘烤时间"])
        reply = ask(model, tok, f"分析一下样本{r['样本ID']}，它的性能是怎么来的？")
        p = parse_mech(reply)
        ok5, _ = chain_structure(reply)
        struct.append(ok5)
        if "ratio" in p and not np.isnan(mech.get("r_phenol_epoxy", np.nan) if r["体系"] != "聚酯金黄" else mech.get("r_nco_oh", np.nan)):
            true_r = mech.get("r_nco_oh" if r["体系"] == "聚酯金黄" else "r_phenol_epoxy", np.nan)
            if not np.isnan(true_r):
                t_ratio += 1
                n_ratio += int(abs(p["ratio"] - true_r) <= 0.15)
        if "ne" in p:
            true_ne = mech.get("ne_effective", np.nan)
            if not np.isnan(true_ne) and true_ne > 0:
                t_ne += 1
                n_ne += int(abs(p["ne"] - true_ne * 1000) / (true_ne * 1000) <= 0.35)
        if "solids" in p:
            true_s = mech.get("solids_frac", np.nan)
            if not np.isnan(true_s):
                t_sol += 1
                n_sol += int(abs(p["solids"] - true_s * 100) <= 6)
        if "tg" in p:
            true_tg = mech.get("tg_fox_solids", np.nan)
            if true_tg is not None and not (isinstance(true_tg, float) and np.isnan(true_tg)):
                t_tg += 1
                n_tg += int(abs(p["tg"] - true_tg) <= 15)
        for sid, rv in re.findall(r"([A-Za-z0-9\-]{3,})（([\d.]+)）实测", reply):
            hit = df[df["样本ID"] == sid]
            if len(hit):
                cite_all += 1
                rr = hit.iloc[0]
                tm = mech_readout({k: float(v) for k, v in rr["组分"].items()},
                                  mlib, rr["烘烤温度"], rr["烘烤时间"])
                tr = tm.get("r_nco_oh" if rr["体系"] == "聚酯金黄" else "r_phenol_epoxy", np.nan)
                if not np.isnan(tr) and abs(float(rv) - tr) <= 0.15:
                    cite_ok += 1
    n = len(cand)
    print(f"\n[A] 思维链正确性（{n}个未参与分析链训练的真实样本，贪心解码）")
    print(f"  链结构五要素(组成/机理/历史/结论/工艺)平均完备 {np.mean(struct):.2f}/5")
    if t_ratio:
        print(f"  当量比正确(±0.15): {n_ratio}/{t_ratio}")
    if t_ne:
        print(f"  交联密度正确(±35%): {n_ne}/{t_ne}")
    if t_sol:
        print(f"  固含正确(±6%): {n_sol}/{t_sol}")
    if t_tg:
        print(f"  共混Tg正确(±15°C): {n_tg}/{t_tg}")
    if cite_all:
        print(f"  历史引用的当量比真实: {cite_ok}/{cite_all}")
    return dict(n=n, struct=float(np.mean(struct)), ratio=(n_ratio, t_ratio),
                ne=(n_ne, t_ne), sol=(n_sol, t_sol), tg=(n_tg, t_tg),
                cite=(cite_ok, cite_all))


# ---------------- 测试B：推理准确性（GBM裁判） ----------------
def gen_fresh_comps(df, mlib, keys, n_per=16):
    rng = random.Random(SEED)
    made, out = set(), []
    rows = df.reset_index(drop=True).to_dict("records")
    tries = 0
    while len(out) < n_per and tries < 800:
        tries += 1
        r = rows[rng.randrange(len(rows))]
        comp = {k: float(v) for k, v in r["组分"].items()}
        total = sum(comp.values())
        mats = sorted(comp)
        c2 = dict(comp)
        for k in rng.sample(mats, min(3, len(mats))):
            delta = rng.uniform(-0.35, 0.45) * comp[k] if comp[k] > 1 else rng.uniform(-0.5, 1.0)
            c2[k] = max(0.0, comp[k] + delta)
        c2 = {k: v for k, v in c2.items() if v > 0.3}
        if not c2:
            continue
        t2 = sum(c2.values())
        c2 = {k: v / t2 * total for k, v in c2.items()}
        key = tuple(sorted((k, round(v, 1)) for k, v in c2.items()))
        if key in made or key in keys:
            continue
        made.add(key)
        bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
        if rng.random() < 0.25:
            bake = (200.0, 10.0) if bake == (205.0, 17.0) else (205.0, 17.0)
        out.append((r["体系"], c2, bake))
    return out


def test_b(model, tok, df, mlib, keys, predictor):
    rng = random.Random(SEED + 1)
    X = build_matrix(df, mlib, predictor.codes).values
    Xs = np.nan_to_num((X - X.mean(0)) / (X.std(0) + 1e-9))
    cases = gen_fresh_comps(df, mlib, keys, 16)
    cov_gbm_t = cov_gbm_m = cov_nn_t = cov_nn_m = got_t = got_m = 0
    widths_t, widths_m = [], []
    ratio_ok = ratio_n = 0
    for system, comp, bake in cases:
        mech = mech_readout(comp, mlib, bake[0], bake[1])
        reply = ask(model, tok, infer_q(system, comp, bake))
        iv = parse_intervals(reply)
        pred = predictor.predict(comp, system, bake[0], bake[1])
        xn = np.nan_to_num(np.array(build_row(comp, system, bake[0], bake[1],
                                              mlib, predictor.codes)).reshape(1, -1))
        d = np.linalg.norm(Xs - xn[0], axis=1)
        nn = df.iloc[np.argsort(d)[:1].tolist()[0]]
        p = parse_mech(reply)
        true_r = mech.get("r_nco_oh" if system == "聚酯金黄" else "r_phenol_epoxy", np.nan)
        if "ratio" in p and not (isinstance(true_r, float) and np.isnan(true_r)):
            ratio_n += 1
            ratio_ok += int(abs(p["ratio"] - true_r) <= 0.15)
        if "t" in iv:
            got_t += 1
            lo, hi = iv["t"]
            widths_t.append((hi - lo) / max((hi + lo) / 2, 1))
            cov_gbm_t += int(lo <= pred["T弯"] <= hi)
            cov_nn_t += int(lo <= nn["T弯"] <= hi)
        if "m" in iv:
            got_m += 1
            lo, hi = iv["m"]
            widths_m.append((hi - lo) / max((hi + lo) / 2, 1))
            cov_gbm_m += int(lo <= pred["MEK"] <= hi)
            cov_nn_m += int(lo <= (nn["MEK"] if not np.isnan(nn["MEK"]) else pred["MEK"]) <= hi)
    print(f"\n[B] 推理准确性（{len(cases)}个全新未训练配方，GBM裁判 + 近邻实测双参照）")
    if got_t:
        print(f"  T弯: 给出区间 {got_t}/{len(cases)}，覆盖GBM裁判 {cov_gbm_t}/{got_t}"
              f"({cov_gbm_t / got_t * 100:.0f}%)，覆盖近邻实测 {cov_nn_t}/{got_t}，"
              f"平均相对宽度 {np.mean(widths_t) * 100:.0f}%")
    if got_m:
        print(f"  MEK: 给出区间 {got_m}/{len(cases)}，覆盖GBM裁判 {cov_gbm_m}/{got_m}"
              f"({cov_gbm_m / got_m * 100:.0f}%)，覆盖近邻实测 {cov_nn_m}/{got_m}，"
              f"平均相对宽度 {np.mean(widths_m) * 100:.0f}%")
    if ratio_n:
        print(f"  新配方机理量-当量比正确(±0.15): {ratio_ok}/{ratio_n}")
    return dict(n=len(cases), t=(cov_gbm_t, got_t, cov_nn_t), m=(cov_gbm_m, got_m, cov_nn_m),
                ratio=(ratio_ok, ratio_n))


# ---------------- 测试C：举一反三 ----------------
def mid(iv):
    return (iv[0] + iv[1]) / 2


def test_c(model, tok, df, mlib):
    res = {}
    # C1 固化剂+50% 方向
    rows = [r for _, r in df.iterrows()
            if any(mlib[k].get("role") == "固化剂" for k in r["组分"])]
    bases = rows[::60][:6]
    mek_ok = t_ok = phrase_ok = 0
    for r in bases:
        comp = {k: float(v) for k, v in r["组分"].items()}
        total = sum(comp.values())
        cure = [k for k in comp if mlib[k].get("role") == "固化剂"]
        c2 = {k: (v * 1.5 if k in cure else v) for k, v in comp.items()}
        s2 = sum(c2.values())
        c2 = {k: v / s2 * total for k, v in c2.items()}
        bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
        a0 = ask(model, tok, infer_q(r["体系"], comp, bake))
        a1 = ask(model, tok, infer_q(r["体系"], c2, bake))
        i0, i1 = parse_intervals(a0), parse_intervals(a1)
        if "m" in i0 and "m" in i1:
            mek_ok += int(mid(i1["m"]) > mid(i0["m"]) * 1.05)
        if "t" in i0 and "t" in i1:
            t_ok += int(mid(i1["t"]) > mid(i0["t"]) * 1.03)
        phrase_ok += int("升高" in a1 or "略降" in a1 or "变脆" in a1 or "变大" in a1)
    res["cure_up"] = (mek_ok, t_ok, phrase_ok, len(bases))
    print(f"\n[C] 举一反三（贪心解码，新造配方对，非训练对）")
    print(f"  固化剂+50%: MEK区间上移 {mek_ok}/{len(bases)}，T弯区间上移 {t_ok}/{len(bases)}，"
          f"文字方向表述正确 {phrase_ok}/{len(bases)}")
    # C1b 固化剂减半
    bases2 = rows[30::90][:3]
    mek_dn = t_dn = 0
    for r in bases2:
        comp = {k: float(v) for k, v in r["组分"].items()}
        total = sum(comp.values())
        cure = [k for k in comp if mlib[k].get("role") == "固化剂"]
        c2 = {k: (v * 0.5 if k in cure else v) for k, v in comp.items()}
        s2 = sum(c2.values())
        c2 = {k: v / s2 * total for k, v in c2.items()}
        bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
        a0 = ask(model, tok, infer_q(r["体系"], comp, bake))
        a1 = ask(model, tok, infer_q(r["体系"], c2, bake))
        i0, i1 = parse_intervals(a0), parse_intervals(a1)
        if "m" in i0 and "m" in i1:
            mek_dn += int(mid(i1["m"]) < mid(i0["m"]) * 0.95)
        if "t" in i0 and "t" in i1:
            t_dn += int(mid(i1["t"]) < mid(i0["t"]) * 0.97)
    res["cure_dn"] = (mek_dn, t_dn, len(bases2))
    print(f"  固化剂减半: MEK区间下移 {mek_dn}/{len(bases2)}，T弯区间下移 {t_dn}/{len(bases2)}")
    # C2 烘烤升级 200×10 → 205×17
    b200 = [r for _, r in df.iterrows()
            if float(r["烘烤温度"]) == 200.0 and float(r["烘烤时间"]) == 10.0]
    b200 = b200[::25][:3]
    mek_up = 0
    for r in b200:
        comp = {k: float(v) for k, v in r["组分"].items()}
        a0 = ask(model, tok, infer_q(r["体系"], comp, (200.0, 10.0)))
        a1 = ask(model, tok, infer_q(r["体系"], comp, (205.0, 17.0)))
        i0, i1 = parse_intervals(a0), parse_intervals(a1)
        if "m" in i0 and "m" in i1:
            mek_up += int(mid(i1["m"]) > mid(i0["m"]) * 1.05)
    res["bake_up"] = (mek_up, len(b200))
    print(f"  烘烤200×10→205×17: MEK区间上移 {mek_up}/{len(b200)}")
    # C3 极端配方坦诚度
    rng = random.Random(SEED + 2)
    resins = [k for k, v in mlib.items() if v.get("role") == "树脂"]
    solvs = [k for k, v in mlib.items() if v.get("role") == "溶剂"]
    cures = [k for k, v in mlib.items() if v.get("role") == "固化剂"]
    ood = []
    for _ in range(4):
        sysname = df.iloc[rng.randrange(len(df))]["体系"]
        kind = rng.randrange(3)
        if kind == 0:
            comp = {rng.choice(solvs): 55.0, rng.choice(cures): 8.0, rng.choice(resins): 30.0}
        elif kind == 1:
            comp = {rng.choice(resins): 40.0, rng.choice(solvs): 10.0, rng.choice(cures): 45.0}
        else:
            comp = {rng.choice(resins): 85.0, rng.choice(solvs): 12.0, rng.choice(cures): 3.0}
        ood.append((sysname, comp, (205.0, 17.0)))
    honest = 0
    for sysname, comp, bake in ood:
        reply = ask(model, tok, infer_q(sysname, comp, bake))
        if ("偏离" in reply or "把握有限" in reply) and not parse_intervals(reply):
            honest += 1
    res["ood"] = (honest, len(ood))
    print(f"  极端配方坦诚(声明偏离/不给数值区间): {honest}/{len(ood)}")
    # C4 低资源体系 聚酯金黄
    polyester = [r for _, r in df.iterrows() if r["体系"] == "聚酯金黄"][:3]
    r_ok = r_n = 0
    for r in polyester:
        comp = {k: float(v) for k, v in r["组分"].items()}
        bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
        reply = ask(model, tok, infer_q("聚酯金黄", comp, bake))
        p = parse_mech(reply)
        mech = mech_readout(comp, mlib, bake[0], bake[1])
        true_n = mech.get("r_nco_oh", np.nan)
        if not np.isnan(true_n):
            r_n += 1
            r_ok += int("ratio" in p and abs(p["ratio"] - true_n) <= 0.15)
    res["polyester"] = (r_ok, r_n, len(polyester))
    print(f"  低资源体系(聚酯金黄): NCO/OH当量比正确(±0.15) {r_ok}/{r_n}")
    return res


def main():
    mlib = mat_lib()
    df = labeled(samples_df())
    trained_ids, infer_qs = trained_registry()
    keys = trained_comp_keys(infer_qs)
    print(f"登记: 分析链训练样本 {len(trained_ids)}，推断训练配方 {len(keys)}")
    model, tok = load_model()
    print("[模型已加载] out/sft/coating_best.pt")

    from predictor import CoatingPredictor
    gp = CoatingPredictor.load()

    test_a(model, tok, df, mlib, trained_ids)
    test_b(model, tok, df, mlib, keys, gp)
    test_c(model, tok, df, mlib)


if __name__ == "__main__":
    main()
