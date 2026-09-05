# -*- coding: utf-8 -*-
"""P0+P1 数据增强：方向对比对 / 模板隔离 / 算术课程 / 双锚教师。

B1 方向对比对：同一近邻锚点下的三元组(基准/固化剂上调/下调)与烘烤、溶剂、蜡配对，
   区间随机理量连续位移(f = 1 − k·δ，δ=当量比相对锚点变化；固化剂↑→当量比↓→交联↑
   →MEK↑/T弯↑，与GBM实证方向一致)。
B2 模板隔离：旧数据中"请推断…性能"问法上挂错模板(方向模板/常规区间模板)的样本剔除，
   "常规区间"模板只保留在改写问法上；推断问法一律绑定完整推理链。
B3 算术课程：当量比与有效交联密度的带步骤计算例，公式与 mech_readout 完全一致。
B4 双锚教师：近邻×2 + GBM点预测(±3mm / ±25%)取并集作为区间，全局信息进入区间。
"""
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "coating_ai"))

from data import mat_lib, samples_df, labeled  # noqa: E402
from features import build_matrix, build_row, mech_readout  # noqa: E402

OUT = os.path.join(HERE, "dataset")
mlib = mat_lib()
rng = random.Random(2027)


# ---------------- 基础工具 ----------------
def role_pct(comp):
    total = sum(comp.values())
    pct = {}
    for k, w in comp.items():
        r = mlib[k].get("role")
        pct[r] = pct.get(r, 0) + w / total * 100
    return pct


def roles_phrase(comp):
    return ", ".join(f"{r}占{role_pct(comp).get(r, 0):.1f}%"
                     for r in ["树脂", "固化剂", "溶剂", "助剂", "颜料"]
                     if role_pct(comp).get(r, 0) > 0.1)


def mech_sentence(mech, system):
    parts = []
    key = "r_nco_oh" if system == "聚酯金黄" else "r_phenol_epoxy"
    v = mech.get(key, 0)
    label = "NCO/OH当量比" if system == "聚酯金黄" else "酚羟基/环氧当量比"
    parts.append(f"{label}约{v:.2f}")
    parts.append(f"有效交联密度量级{mech.get('ne_effective', 0) * 1000:.1f}e-3 mol/g")
    parts.append(f"固含{mech.get('solids_frac', 0) * 100:.0f}%")
    tg = mech.get("tg_fox_solids")
    if tg is not None and not (isinstance(tg, float) and np.isnan(tg)):
        parts.append(f"共混Tg约{tg:.0f}°C")
    return "，".join(parts)


def perf_str(r):
    seg = []
    if r is not None and not np.isnan(r["T弯"]):
        seg.append(f"T弯{r['T弯']:.1f}mm")
    if r is not None and not np.isnan(r["MEK"]):
        seg.append(f"MEK{r['MEK']:.0f}次")
    if r is not None and not np.isnan(r["水煮"]):
        seg.append(f"水煮{r['水煮']:.0f}级")
    return "、".join(seg)


def comp_str(comp):
    return "、".join(f"{k} {v:.1f}%" for k, v in sorted(comp.items(), key=lambda x: -x[1]))


def infer_q(system, comp, bake):
    return (f"请推断这个新配方的性能：{system}体系，{comp_str(comp)}，"
            f"烘烤{bake[0]:.0f}°C×{bake[1]:.0f}min。")


def ratio_of(mech, system):
    return mech.get("r_nco_oh" if system == "聚酯金黄" else "r_phenol_epoxy", np.nan)


def f_shift(delta):
    f_mek = float(np.clip(1.0 + 1.2 * delta, 0.62, 1.58))
    f_t = float(np.clip(1.0 + 0.4 * delta, 0.70, 1.42))
    return f_mek, f_t


def interval(m0, t0, f_mek, f_t):
    mm, tm = m0 * f_mek, t0 * f_t
    mek = (max(2, int(round(mm * 0.78))), max(3, int(round(mm * 1.30))))
    tb = (max(0, int(round(tm * 0.85))), max(1, int(round(tm * 1.18))))
    if mek[1] <= mek[0]:
        mek = (mek[0], mek[0] + 2)
    if tb[1] <= tb[0]:
        tb = (tb[0], tb[0] + 1)
    return mek, tb


def direction_phrase(delta):
    if delta > 0.03:
        return "本配方固化剂相对参照偏多，交联密度上升"
    if delta < -0.03:
        return "本配方固化剂相对参照偏少，交联不足交联偏弱"
    return "本配方固化剂比例与参照相近，交联程度与参照相当"


def t_qualifier(f_t):
    if f_t > 1.02:
        return "（柔韧性略降）"
    if f_t < 0.98:
        return "（柔韧性改善）"
    return "（柔韧性尚可）"


def build_answer(system, comp, bake, anchor, mech, delta, extra_m=1.0, extra_t=1.0,
                 extra_phrase="", shuizhu=None, cite2=None):
    m0 = float(anchor["MEK"]) if not np.isnan(anchor["MEK"]) else 80.0
    t0 = float(anchor["T弯"]) if not np.isnan(anchor["T弯"]) else 18.0
    n0 = int(anchor["水煮"]) if not np.isnan(anchor["水煮"]) else 3
    f_mek, f_t = f_shift(delta)
    f_mek *= extra_m
    f_t *= extra_t
    (m_lo, m_hi), (t_lo, t_hi) = interval(m0, t0, f_mek, f_t)
    n = shuizhu if shuizhu is not None else n0
    s = [f"我按机理来推断这个{system}体系新配方：{roles_phrase(comp)}，"
         f"{mech_sentence(mech, system)}；",
         f"历史最近配方{anchor['样本ID']}（{comp_str({k: float(v) for k, v in anchor['组分'].items()})[:60]}…）"
         f"实测{perf_str(anchor)}；"]
    if cite2 is not None:
        s.append(f"次近邻{cite2['样本ID']}实测{perf_str(cite2)}；")
    s.append(f"{direction_phrase(delta)}{extra_phrase}，"
             f"据此推断：T弯约{t_lo}~{t_hi}mm{t_qualifier(f_t)}，"
             f"MEK擦拭约{m_lo}~{m_hi}次，水煮大概率{n}级；")
    advice = []
    if f_t > 1.02:
        advice.append("若嫌脆可将固化剂降到12~15%")
    if m_lo < 60:
        advice.append("耐溶剂余量不足时可提高固化剂占比或延长烘烤")
    s.append(("调整方向：" + "；".join(advice) + "。") if advice
             else "整体在体系常规工作区内，可按原工艺试板验证。")
    return "".join(s)


def dose_variant(comp, keys, mult):
    c2 = {k: (v * mult if k in keys else v) for k, v in comp.items()}
    tot = sum(c2.values())
    return {k: v / tot * sum(comp.values()) for k, v in c2.items()}


def cure_frac(comp):
    total = sum(comp.values())
    return sum(v for k, v in comp.items() if mlib[k].get("role") == "固化剂") / max(total, 1e-9)


def dose_delta(comp, anchor):
    """固化剂剂量相对参照的对数变化：固化剂↑ → δ↑ → 交联↑ → MEK↑/T弯↑。"""
    fa = cure_frac({k: float(v) for k, v in anchor["组分"].items()})
    fb = cure_frac(comp)
    if fa < 0.005 or fb < 0.005:
        return np.nan
    return float(np.log(fb / fa))


# ---------------- 近邻索引 ----------------
def build_nn_index(df):
    codes_df = df
    X = build_matrix(codes_df, mlib, sorted({c for _, r in df.iterrows() for c in r["组分"]})).values
    Xs = np.nan_to_num((X - X.mean(0)) / (X.std(0) + 1e-9))
    idx_by_sys = {s: np.where(df["体系"].values == s)[0] for s in df["体系"].unique()}

    def nn_same_system(row_i, k=2):
        xi = Xs[row_i]
        cand = idx_by_sys[df.iloc[row_i]["体系"]]
        d = np.linalg.norm(Xs[cand] - xi, axis=1)
        order = np.argsort(d)
        return [int(cand[j]) for j in order if cand[j] != row_i][:k]
    return nn_same_system


def valid_anchor(row):
    return (not np.isnan(row["MEK"]) and row["MEK"] > 0 and not np.isnan(row["T弯"])
            and not np.isnan(ratio_of(mech_readout(
                {k: float(v) for k, v in row["组分"].items()}, mlib,
                row["烘烤温度"], row["烘烤时间"]), row["体系"])))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = labeled(samples_df()).reset_index(drop=True)
    nn_of = build_nn_index(df)
    rows = []

    def add(q, a, w=1):
        for _ in range(w):
            rows.append({"conversations": [{"role": "user", "content": q},
                                           {"role": "assistant", "content": a}]})

    used_keys = set()

    def key_of(comp):
        return tuple(sorted((k, round(v, 1)) for k, v in comp.items()))

    # ---------- B1 固化剂三元组 ----------
    n_cure = 0
    order = list(range(len(df)))
    rng.shuffle(order)
    for i in order:
        if n_cure >= 200:
            break
        r = df.iloc[i]
        if not valid_anchor(r):
            continue
        comp = {k: float(v) for k, v in r["组分"].items()}
        cure = [k for k in comp if mlib[k].get("role") == "固化剂"]
        if not cure:
            continue
        nn_i = nn_of(i, 4)
        anchor = next((df.iloc[j] for j in nn_i if valid_anchor(df.iloc[j])), None)
        if anchor is None:
            continue
        system = r["体系"]
        bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
        variants = {}
        ok = True
        for tag, mult in (("base", 1.0), ("up", 1.3), ("dn", 0.6)):
            c2 = comp if tag == "base" else dose_variant(comp, cure, mult)
            mech = mech_readout(c2, mlib, bake[0], bake[1])
            dv = dose_delta(c2, anchor)
            if np.isnan(dv):
                ok = False
                break
            variants[tag] = (c2, mech, dv)
        if not ok:
            continue
        d_base, d_up, d_dn = variants["base"][2], variants["up"][2], variants["dn"][2]
        f_b = f_shift(d_base)[0]
        f_u, f_d = f_shift(d_up)[0], f_shift(d_dn)[0]
        if f_u < f_b + 0.04 or f_d > f_b - 0.04:
            continue
        for tag in ("base", "up", "dn"):
            c2, mech, delta = variants[tag]
            k = key_of(c2)
            if k in used_keys:
                ok = False
                break
            used_keys.add(k)
        if not ok:
            continue
        n0 = int(anchor["水煮"]) if not np.isnan(anchor["水煮"]) else 3
        for tag in ("base", "up", "dn"):
            c2, mech, delta = variants[tag]
            shz = n0 if tag == "base" else (min(5, max(4, n0)) if tag == "up"
                                            else max(2, min(n0, 3)))
            add(infer_q(system, c2, bake),
                build_answer(system, c2, bake, anchor, mech, delta, shuizhu=shz))
        n_cure += 1
    print(f"B1 固化剂三元组: {n_cure} 组 = {n_cure * 3} 条")

    # ---------- B1 烘烤升级对 ----------
    n_bake = 0
    for i in order:
        if n_bake >= 120:
            break
        r = df.iloc[i]
        if not valid_anchor(r):
            continue
        if (float(r["烘烤温度"]), float(r["烘烤时间"])) != (200.0, 10.0):
            continue
        comp = {k: float(v) for k, v in r["组分"].items()}
        system = r["体系"]
        nn_i = nn_of(i, 4)
        anchor = next((df.iloc[j] for j in nn_i if valid_anchor(df.iloc[j])), None)
        if anchor is None:
            continue
        mech0 = mech_readout(comp, mlib, 200.0, 10.0)
        mech1 = mech_readout(comp, mlib, 205.0, 17.0)
        delta = dose_delta(comp, anchor)
        k = key_of(comp)
        if k in used_keys or np.isnan(delta):
            continue
        used_keys.add(k)
        add(infer_q(system, comp, (200.0, 10.0)),
            build_answer(system, comp, (200.0, 10.0), anchor, mech0, delta))
        add(infer_q(system, comp, (205.0, 17.0)),
            build_answer(system, comp, (205.0, 17.0), anchor, mech1, delta,
                         extra_m=1.28, extra_t=1.06,
                         extra_phrase="；改为205°C×17min后等效固化强度明显提升",
                         shuizhu=min(5, max(4, int(anchor["水煮"]) if not np.isnan(anchor["水煮"]) else 3))))
        n_bake += 1
    print(f"B1 烘烤升级对: {n_bake} 组 = {n_bake * 2} 条")

    # ---------- B1 溶剂/蜡配对 ----------
    n_solv = n_wax = 0
    for i in order:
        r = df.iloc[i]
        if not valid_anchor(r):
            continue
        comp = {k: float(v) for k, v in r["组分"].items()}
        system = r["体系"]
        bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
        nn_i = nn_of(i, 4)
        anchor = next((df.iloc[j] for j in nn_i if valid_anchor(df.iloc[j])), None)
        if anchor is None:
            continue
        solv = [k for k in comp if mlib[k].get("role") == "溶剂"]
        wax = [k for k in comp if float(mlib[k].get("wax") or 0) > 0]
        if solv and n_solv < 100:
            c2 = dose_variant(comp, solv, 4.0 / 3.0)
            k = key_of(c2)
            if k not in used_keys:
                used_keys.add(k)
                mech = mech_readout(c2, mlib, bake[0], bake[1])
                delta = dose_delta(c2, anchor)
                if not np.isnan(delta):
                    add(infer_q(system, c2, bake),
                        build_answer(system, c2, bake, anchor, mech, delta,
                                     extra_m=0.92, extra_t=1.0,
                                     extra_phrase="；溶剂比例提高后固含下降，耐溶剂与水煮上限受影响"))
                    n_solv += 1
        if wax and n_wax < 80:
            c2 = dose_variant(comp, wax, 2.0)
            k = key_of(c2)
            if k not in used_keys:
                used_keys.add(k)
                mech = mech_readout(c2, mlib, bake[0], bake[1])
                delta = dose_delta(c2, anchor)
                if not np.isnan(delta):
                    add(infer_q(system, c2, bake),
                        build_answer(system, c2, bake, anchor, mech, delta,
                                     extra_phrase="；蜡粉加倍主要影响滑爽抗划与外观，对交联类指标影响很小"))
                    n_wax += 1
    print(f"B1 溶剂对: {n_solv} 组，蜡对: {n_wax} 组")

    # ---------- B4 双锚教师 ----------
    from predictor import CoatingPredictor
    gp = CoatingPredictor.load()
    n_dual = 0
    tried = 0
    rng2 = random.Random(2028)
    while n_dual < 500 and tried < 4000:
        tried += 1
        r = df.iloc[rng2.randrange(len(df))]
        if not valid_anchor(r):
            continue
        comp = {k: float(v) for k, v in r["组分"].items()}
        total = sum(comp.values())
        mats = sorted(comp)
        c2 = dict(comp)
        for k in rng2.sample(mats, min(3, len(mats))):
            d = rng2.uniform(-0.35, 0.45) * comp[k] if comp[k] > 1 else rng2.uniform(-0.5, 1.0)
            c2[k] = max(0.0, comp[k] + d)
        c2 = {k: v for k, v in c2.items() if v > 0.3}
        if not c2:
            continue
        t2 = sum(c2.values())
        c2 = {k: v / t2 * total for k, v in c2.items()}
        k = key_of(c2)
        if k in used_keys:
            continue
        used_keys.add(k)
        system = r["体系"]
        bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
        i_row = int(r.name)
        nn_i = nn_of(i_row, 3)
        anchors = [df.iloc[j] for j in nn_i if valid_anchor(df.iloc[j])]
        if len(anchors) < 2:
            continue
        a1, a2 = anchors[0], anchors[1]
        mech = mech_readout(c2, mlib, bake[0], bake[1])
        delta = dose_delta(c2, a1)
        if np.isnan(delta):
            continue
        pred = gp.predict(c2, system, bake[0], bake[1])
        m0 = float(a1["MEK"]) if not np.isnan(a1["MEK"]) else 80.0
        t0 = float(a1["T弯"]) if not np.isnan(a1["T弯"]) else 18.0
        f_mek, f_t = f_shift(delta)
        (m_lo, m_hi), (t_lo, t_hi) = interval(m0, t0, f_mek, f_t)
        m_lo = max(2, int(min(m_lo, pred["MEK"] * 0.75)))
        m_hi = int(max(m_hi, pred["MEK"] * 1.25))
        t_lo = max(0, int(min(t_lo, pred["T弯"] - 3)))
        t_hi = int(max(t_hi, pred["T弯"] + 3))
        n1 = int(a1["水煮"]) if not np.isnan(a1["水煮"]) else 3
        n2 = int(a2["水煮"]) if not np.isnan(a2["水煮"]) else 3
        s = [f"我按机理来推断这个{system}体系新配方：{roles_phrase(c2)}，"
             f"{mech_sentence(mech, system)}；",
             f"历史相近配方{a1['样本ID']}（{comp_str({k2: float(v) for k2, v in a1['组分'].items()})[:60]}…）"
             f"实测{perf_str(a1)}，次近邻{a2['样本ID']}实测{perf_str(a2)}；",
             f"{direction_phrase(delta)}，据此推断：T弯约{t_lo}~{t_hi}mm{t_qualifier(f_t)}，"
             f"MEK擦拭约{m_lo}~{m_hi}次，水煮大概率{min(5, max(3, n1, n2))}级；",
             "区间已结合机理方向与全局模型校准，建议按标准工艺试板验证。"]
        add(infer_q(system, c2, bake), "".join(s))
        n_dual += 1
    print(f"B4 双锚教师: {n_dual} 条")

    # ---------- B3 算术课程 ----------
    LIT_NCO = {"RY460": 797.0, "RY075N": 606.0}
    LIT_AMINE = {"RA009": 416.7, "RA083": 301.5, "RA824": 210.5}
    n_k1 = n_k2 = 0
    rng3 = random.Random(2029)
    k1_counts = {}
    tried_k1 = 0
    while n_k1 < 600 and tried_k1 < 20000:
        tried_k1 += 1
        r = df.iloc[rng3.randrange(len(df))]
        comp = {k: float(v) for k, v in r["组分"].items()}
        system = r["体系"]
        bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
        mech = mech_readout(comp, mlib, bake[0], bake[1])
        ck = key_of(comp)
        cap = 50 if system == "聚酯金黄" else 2
        if k1_counts.get(ck, 0) >= cap:
            continue
        if system == "聚酯金黄":
            eq_nco = sum(a / LIT_NCO[k] for k, a in comp.items()
                         if k in LIT_NCO)
            eq_oh = sum(a * float(mlib[k].get("fg_oh") or 0) / 100.0
                        for k, a in comp.items())
            if eq_nco <= 1e-6 or eq_oh <= 1e-6:
                continue
            rv = mech.get("r_nco_oh", 0.0)
            if rv <= 0 or abs(eq_nco / max(eq_oh, 1e-9) - rv) > 0.01:
                continue
            first_nco = next((k for k in comp if k in LIT_NCO), None)
            first_oh = next((k for k, a in comp.items()
                             if float(mlib[k].get("fg_oh") or 0) > 0), None)
            seg = [f"主要NCO来源{first_nco}：{comp[first_nco]:.1f}/{LIT_NCO[first_nco]:.0f}"
                   f"={comp[first_nco] / LIT_NCO[first_nco]:.3f}mol",
                   f"主要羟基来源{first_oh}：{comp[first_oh]:.1f}×"
                   f"{float(mlib[first_oh].get('fg_oh') or 0):.3f}/100="
                   f"{comp[first_oh] * float(mlib[first_oh].get('fg_oh') or 0) / 100.0:.3f}mol",
                   f"全部组分加总：NCO合计{eq_nco:.3f}mol、羟基合计{eq_oh:.3f}mol",
                   f"NCO/OH当量比={eq_nco:.3f}÷{eq_oh:.3f}≈{rv:.2f}，即该配方的当量比指标。"]
            q = (f"算一下这个{system}配方当量比：{comp_str(comp)}，"
                 f"烘烤{bake[0]:.0f}°C×{bake[1]:.0f}min。")
            add(q, "；".join(seg))
            n_k1 += 1
        else:
            eq_ep = sum(a * float(mlib[k].get("fg_epoxy") or 0) / 100.0
                        for k, a in comp.items())
            act_h = 0.0
            first_ep = first_ph = None
            for k, a in comp.items():
                m = mlib[k]
                fg_oh = float(m.get("fg_oh") or 0)
                contrib = fg_oh / 100.0 * a
                if m.get("rtype") == "酚醛" and m.get("role") == "固化剂":
                    if first_ph is None and fg_oh > 0:
                        first_ph = (k, a, fg_oh)
                if float(m.get("fg_epoxy") or 0) > 0 and first_ep is None:
                    first_ep = (k, a, float(m.get("fg_epoxy")))
                if m.get("rtype") in ("聚酯", "丙烯酸", "氨基", "环氧", "酚醛"):
                    act_h += contrib
                am = LIT_AMINE.get(k)
                if am:
                    act_h += a / am
                act_h += float(m.get("fg_cooh") or 0) / 100.0 * a
            if eq_ep <= 1e-6 or act_h <= 1e-6:
                continue
            rv = mech.get("r_phenol_epoxy", 0.0)
            if rv <= 0 or abs(eq_ep / act_h - rv) > 0.02:
                continue
            seg = []
            if first_ep is not None:
                k, a2, fg = first_ep
                seg.append(f"主要环氧来源{k}：{a2:.1f}×{fg:.3f}/100={a2 * fg / 100.0:.3f}mol")
            if first_ph is not None:
                k, a2, fg = first_ph
                seg.append(f"主要酚羟基来源{k}：{a2:.1f}×{fg:.3f}/100={a2 * fg / 100.0:.3f}mol")
            seg.append(f"全部组分加总：环氧合计{eq_ep:.3f}mol、活性氢合计{act_h:.3f}mol")
            seg.append(f"当量比=环氧{eq_ep:.3f}÷活性氢{act_h:.3f}≈{rv:.2f}，即该配方的当量比指标。")
            q = (f"算一下这个{system}配方当量比：{comp_str(comp)}，"
                 f"烘烤{bake[0]:.0f}°C×{bake[1]:.0f}min。")
            add(q, "；".join(seg))
            n_k1 += 1
            k1_counts[ck] = k1_counts.get(ck, 0) + 1
    print(f"B3 当量比算术: {n_k1} 条")

    ALPHA = {(200.0, 10.0): 0.28, (205.0, 17.0): 0.51}
    k2_counts = {}
    tried_k2 = 0
    while n_k2 < 600 and tried_k2 < 20000:
        tried_k2 += 1
        r = df.iloc[rng3.randrange(len(df))]
        comp = {k: float(v) for k, v in r["组分"].items()}
        system = r["体系"]
        bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
        mech = mech_readout(comp, mlib, bake[0], bake[1])
        ck = key_of(comp)
        npot = mech.get("ne_potential", 0.0)
        ne = mech.get("ne_effective", 0.0)
        if npot * 1000 < 0.2 or ne <= 0:
            continue
        alpha = ALPHA.get(bake)
        if alpha is None:
            continue
        q = (f"估计这个{system}配方的有效交联密度：{comp_str(comp)}，"
             f"烘烤{bake[0]:.0f}°C×{bake[1]:.0f}min。")
        a = (f"限制性当量与网络官能度给出潜在交联密度约{npot * 1000:.1f}e-3 mol/g；"
             f"{bake[0]:.0f}°C×{bake[1]:.0f}min下转化率α≈{alpha:.2f}，"
             f"有效交联密度={npot * 1000:.1f}×{alpha:.2f}≈{ne * 1000:.1f}e-3 mol/g。")
        if abs(npot * alpha - ne) / max(ne, 1e-9) > 0.05:
            continue
        if k2_counts.get(ck, 0) >= 3:
            continue
        k2_counts[ck] = k2_counts.get(ck, 0) + 1
        add(q, a)
        n_k2 += 1
    print(f"B3 交联密度算术: {n_k2} 条")

    with open(os.path.join(OUT, "sft_upgrade.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"新增SFT合计: {len(rows)} 条 -> sft_upgrade.jsonl")

    # ---------- B2 模板隔离：过滤旧数据 ----------
    kept, dropped = [], 0
    for name in ("sft_coating.jsonl", "sft_refine.jsonl"):
        p = os.path.join(OUT, name)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            conv = json.loads(line)["conversations"]
            u, a = conv[0]["content"], conv[1]["content"]
            if (u.startswith("请推断这个新配方的性能")
                    and ("据此推断" in a or "区间把握度中等" in a)):
                dropped += 1
                continue
            kept.append({"conversations": conv})
    print(f"B2 模板隔离: 剔除 {dropped} 条，保留旧数据 {len(kept)} 条")

    all_rows = kept + rows
    random.Random(7).shuffle(all_rows)
    with open(os.path.join(OUT, "sft_upgrade_all.jsonl"), "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"合并SFT: {len(all_rows)} 条 -> sft_upgrade_all.jsonl")


if __name__ == "__main__":
    main()
