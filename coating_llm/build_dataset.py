# -*- coding: utf-8 -*-
"""专家推理链训练数据生成。

思路：把 coating_ai 的机理量（当量比/交联密度/Tg/固化裕度）与历史近邻检索
当作"教师"，为每个样本写一条有据可依的推理链；再对真实配方做受控扰动生成
"新配方"，答案由近邻实测值 + 机理方向合成——区间、置信度、调整建议齐全，
复现真专家"看配方 → 讲机理 → 引案例 → 下判断"的推断方式。
"""
import json
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "coating_ai"))
TDS_DIR = os.path.join(ROOT, "3NEW", "generalization", "TDS-SDS")
PKL = os.path.join(ROOT, "3NEW", "generalization", "data", "merged_data.pkl")

from data import mat_lib, samples_df, labeled, present_codes  # noqa: E402
from features import build_matrix, mech_readout, build_row  # noqa: E402

OUT_DIR = os.path.join(HERE, "dataset")
rng = random.Random(2025)
mlib = mat_lib()

SYSTEM_PROFILE = {
    "环氧酚醛": ("环氧树脂提供附着力与柔韧性，酚醛固化剂提供高交联密度与耐蒸煮性，"
               "是罐头内涂的经典体系，烘烤200°C×10min或205°C×17min完全固化"),
    "环氧配比方案": ("多树脂配比实验体系，含色浆与促进剂，考察不同树脂/固化剂配比"
                 "对力学与耐性的影响，烘烤205°C×17min"),
    "聚酯金黄": ("聚酯树脂为主、封端异氰酸酯(RY系)固化的粉末/卷材类体系，"
              "羟基与NCO计量比决定交联程度，烘烤205°C×17min"),
}

MECH_KNOWLEDGE = [
    "环氧树脂中的环氧基与酚醛树脂的酚羟基在高温下发生逐步加成反应，形成以C-N、C-C键连接的三维交联网络。交联点密度越高，涂膜抵抗溶剂溶胀的能力越强，表现为MEK擦拭次数越高；同时网络刚性强，链段活动性下降，涂膜变硬变脆，T弯数值变大。",
    "化学计量比是配方设计的核心：酚羟基当量与环氧当量之比r_phenol_epoxy接近体系经验区间时固化最充分。固化剂过量则残留小分子使涂膜发脆并可能渗出；不足则交联不完全，MEK擦拭明显偏低。",
    "烘烤窗口决定转化率：温度提供反应活化能，时间保证扩散与反应完成。200°C×10min与205°C×17min是环氧酚醛体系两条经验制度，温度每差10°C等效反应时间约差一倍，工艺不足时即使配方合理也会欠固化。",
    "涂膜柔韧性取决于交联密度与树脂链段柔性之间的平衡。柔韧环氧骨干与较低的交联密度带来小T弯值；高官能度固化剂、高Tg树脂都推高交联密度，使涂膜耐溶剂性上升而柔韧性下降。",
    "水煮性能考察涂膜在沸水中的水解稳定性：交联密度高、极性基团残留少、固化完全的涂膜不易起泡脱落。水煮等级与MEK擦拭通常正相关，因为两者都依赖网络的致密程度。",
    "溶剂的作用贯穿施工与成膜：溶解树脂、调节粘度、控制挥发梯度。沸程与 Hansen 溶解度参数匹配的混合溶剂能得到无缩孔的连续涂膜；挥发过快易导致橘皮，过慢则降低固化初期的表干速度。",
    "助剂用量虽小作用关键：流平剂降低表面张力差消除橘皮；蜡粉提供滑爽与抗划伤；酸性催化剂加速环氧-酚醛反应缩短固化窗口。比例失调会带来缩孔、雾影或层间附着力问题。",
    "Fox方程描述共混树脂的玻璃化转变温度：按组分体积分数的倒数加权。配方Tg与烘烤温度的相对关系决定固化时是否进入玻璃化受限区，过高的Tg限制链段运动，转化率被冻结在较低水平。",
    "分析一个配方要按固定流程：先看树脂与固化剂的主体构成与比例，再算活性当量比与交联密度量级，然后对照烘烤工艺判断转化率，最后检索历史相近配方的实测结果作为参照，给出区间化的性能推断。",
    "聚酯-异氰酸酯体系中，羟基与NCO的当量比r_nco_oh控制交联程度：接近1时网络最完整，MEK与水煮都处于高位；明显偏离时残留未反应基团，耐性下降。封端异氰酸酯需要在解封温度以上烘烤才能参与反应。",
]

IDENTITY_QA = [
    ("你是谁？", "我是涂釉，涂料领域的配方性能小专家。我熟悉环氧酚醛、环氧配比方案、聚酯金黄三个体系的配方与实测性能，掌握69种原料的TDS/SDS档案，能像配方工程师一样分析组成、讲清机理、对照历史数据来推断新配方的性能。"),
    ("你叫什么名字？", "我叫涂釉。涂是涂料的涂，釉是釉料的釉。把我当成一位随时在线的涂料配方工程师就好。"),
    ("你能做什么？", "我可以：1)按机理分析配方组成并推断T弯、MEK擦拭、水煮性能；2)介绍原料档案与TDS/SDS要点；3)解释检测指标与固化机理；4)给出性能调优的方向与代价。"),
    ("你怎么推断新配方的性能？", "我的流程与配方工程师一致：先看树脂/固化剂/溶剂/助剂构成与比例，再算活性当量比和交联密度量级，结合烘烤工艺判断固化程度，最后对照历史相近配方的实测结果，给出区间化的推断和调整建议。"),
    ("你的推断可靠吗？", "我的判断来自项目421条实测数据加TDS/SDS化学知识。配方落在历史数据覆盖范围内时，推断区间比较可信；偏离过大时我会明说把握有限，只给方向性判断。"),
    ("什么是T弯？", "T弯是评价涂膜柔韧性的指标：将样板绕标准轴对折弯曲，观察涂膜开裂情况，记录不开裂的最小厚度档。数值越小，柔韧性越好。本项目实测范围0~60mm，主体集中在14~22mm。"),
    ("什么是MEK擦拭？", "MEK擦拭按ASTM D5402执行：蘸丁酮的棉布往复擦拭涂膜直至磨穿，记录次数。次数越高交联密度越大、耐溶剂性越好。本项目实测2~550次，300次以上通常视为高度固化。"),
    ("什么是水煮？", "水煮考察涂膜耐沸水性：样板在沸水中煮规定时间后检查起泡、变色、附着力，按严重程度评级。本项目为1~5级，4级以上为耐水煮优秀。"),
]


def fmt_comp(comp):
    items = sorted(comp.items(), key=lambda x: -x[1])
    return "、".join(f"{k}({mlib[k].get('role')}){v:.1f}%" for k, v in items)


def role_pct(comp):
    total = sum(comp.values())
    pct = {}
    for k, w in comp.items():
        r = mlib[k].get("role")
        pct[r] = pct.get(r, 0) + w / total * 100
    return pct


def mech_sentence(mech, system):
    parts = []
    if system in ("环氧酚醛", "环氧配比方案"):
        r = mech.get("r_phenol_epoxy", 0)
        parts.append(f"酚羟基/环氧当量比约{r:.2f}")
    if system == "聚酯金黄":
        n = mech.get("r_nco_oh", 0)
        parts.append(f"NCO/OH当量比约{n:.2f}")
    parts.append(f"有效交联密度量级{mech.get('ne_effective', 0) * 1000:.1f}e-3 mol/g")
    parts.append(f"固含{mech.get('solids_frac', 0) * 100:.0f}%")
    tg = mech.get("tg_fox_solids")
    if tg is not None and not np.isnan(tg):
        parts.append(f"共混Tg约{tg:.0f}°C")
    return "，".join(parts)


def perf_str(y_t, y_m, y_w):
    seg = []
    if y_t is not None and not np.isnan(y_t):
        seg.append(f"T弯{y_t:.1f}mm")
    if y_m is not None and not np.isnan(y_m):
        seg.append(f"MEK{y_m:.0f}次")
    if y_w is not None and not np.isnan(y_w):
        seg.append(f"水煮{y_w:.0f}级")
    return "、".join(seg)


def build_nn(df):
    X = build_matrix(df, mlib, present_codes(df)).values
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xs = np.nan_to_num(Xs)

    def nn(idx, k=2):
        d = np.linalg.norm(Xs - Xs[idx], axis=1)
        d[idx] = np.inf
        return df.iloc[np.argsort(d)[:k]].index.tolist()
    return nn


def analysis_answer(row, nn_rows, system):
    comp = {k: float(v) for k, v in row["组分"].items()}
    mech = mech_readout(comp, mlib, row["烘烤温度"], row["烘烤时间"])
    pct = role_pct(comp)
    s = [f"我按固定流程分析样本{row['样本ID']}：",
         f"组成上{', '.join(f'{r}占{pct.get(r, 0):.1f}%' for r in ['树脂', '固化剂', '溶剂', '助剂', '颜料'] if pct.get(r, 0) > 0.1)}",
         f"，机理量{mech_sentence(mech, system)}；"]
    if nn_rows is not None and len(nn_rows):
        refs = "；".join(
            f"{r['样本ID']}（{mech_readout({k: float(v) for k, v in r['组分'].items()}, mlib, r['烘烤温度'], r['烘烤时间']).get('r_phenol_epoxy', 0):.2f}）实测{perf_str(r['T弯'], r['MEK'], r['水煮'])}"
            for _, r in nn_rows.iterrows())
        s.append(f"对照历史相近配方：{refs}。")
    s.append(f"固化与性能模式符合体系规律，实测{perf_str(row['T弯'], row['MEK'], row['水煮'])}，"
             f"烘烤{row['烘烤温度']:.0f}°C×{row['烘烤时间']:.0f}min。")
    return "".join(s)


def direction_shift(base, d_mek, d_t):
    """按机理方向把近邻基线推成区间。d_mek/d_t ∈ {-1,0,+1}"""
    if d_mek > 0:
        mek = (base[0] * 0.85, base[0] * 1.4)
    elif d_mek < 0:
        mek = (base[0] * 0.6, base[0] * 1.05)
    else:
        mek = (base[0] * 0.85, base[0] * 1.2)
    if d_t > 0:
        tb = (base[1] * 0.95, base[1] * 1.3)
    elif d_t < 0:
        tb = (base[1] * 0.7, base[1] * 1.05)
    else:
        tb = (base[1] * 0.85, base[1] * 1.15)
    return (min(mek), max(mek)), (min(tb), max(tb))


def infer_answer(new_comp, system, bake, nn_rows, ref_mech, new_mech):
    pct = role_pct(new_comp)
    s = [f"我按机理来推断这个{system}体系新配方：",
         f"{', '.join(f'{r}占{pct.get(r, 0):.1f}%' for r in ['树脂', '固化剂', '溶剂', '助剂', '颜料'] if pct.get(r, 0) > 0.1)}",
         f"，{mech_sentence(new_mech, system)}；"]
    if nn_rows is None or not len(nn_rows):
        s.append("该组成偏离项目历史数据范围较大，我只能给方向性判断："
                 "参照同类体系机理，固化网络应当可以形成，但具体数值把握有限，建议实测试做。")
        return "".join(s)
    r0 = nn_rows.iloc[0]
    base = (r0["MEK"] if not np.isnan(r0["MEK"]) else 80,
            r0["T弯"] if not np.isnan(r0["T弯"]) else 18)
    dr = (new_mech.get("r_phenol_epoxy", 0)
          - mech_readout({k: float(v) for k, v in r0["组分"].items()}, mlib,
                         r0["烘烤温度"], r0["烘烤时间"]).get("r_phenol_epoxy", 0))
    d_mek = int(np.sign(dr))
    d_t = d_mek
    (mek_lo, mek_hi), (t_lo, t_hi) = direction_shift(base, d_mek, d_t)
    s.append(f"历史最近配方{r0['样本ID']}（{fmt_comp({k: float(v) for k, v in r0['组分'].items()})[:60]}…）"
             f"实测{perf_str(r0['T弯'], r0['MEK'], r0['水煮'])}；")
    s.append(f"本配方当量比相对参照{'升高' if d_mek > 0 else ('降低' if d_mek < 0 else '相近')}，"
             f"据此推断：T弯约{max(t_lo, 0):.0f}~{t_hi:.0f}mm（柔韧性{'略降' if d_t > 0 else '尚可'}），"
             f"MEK擦拭约{mek_lo:.0f}~{mek_hi:.0f}次，水煮大概率{4 if base and r0['水煮'] >= 4 or r0['水煮'] >= 3 else 3}级；")
    advice = []
    if d_t > 0:
        advice.append("若嫌脆可将固化剂降到12~15%")
    if mek_lo < 60:
        advice.append("耐溶剂余量不足时可提高固化剂占比或延长烘烤")
    s.append(("调整方向：" + "；".join(advice) + "。") if advice
             else "整体在体系常规工作区内，可按原工艺试板验证。")
    return "".join(s)


def counterfactual_qa(comp, system, bake):
    """反事实：改变某类组成，问性能走向。"""
    pct = role_pct(comp)
    rules = []
    cure = [k for k in comp if mlib[k].get("role") == "固化剂"]
    if cure:
        rules.append((f"把固化剂（{'、'.join(cure)}）比例提高约50%",
                      "交联密度上升，MEK擦拭次数会明显提高，水煮等级通常随之上升；代价是涂膜变硬变脆，T弯数值变大，柔韧性下降。"))
        rules.append((f"把固化剂比例降到约一半",
                      "交联不足，MEK擦拭次数会大幅下降（可能掉到几十次的量级），水煮等级也难以维持；好处是涂膜更柔韧，T弯变小。"))
    solv = [k for k in comp if mlib[k].get("role") == "溶剂"]
    if solv:
        rules.append(("把溶剂比例提高约三分之一",
                      "固含下降，施工粘度更低、流平更好，但挥发物增多，涂膜致密度略降，耐溶剂与水煮的上限会受影响，烘烤负荷也更大。"))
    wax = [k for k in comp if "蜡" in k]
    if wax:
        rules.append(("把蜡粉比例提高一倍",
                      "滑爽性和抗划伤会改善，但过量蜡会降低层间附着并在烘烤中上浮产生雾影，需要控制在一个较小的比例窗口内。"))
    if system == "环氧酚醛":
        rules.append(("把烘烤从200°C×10min改成205°C×17min",
                      "等效固化强度明显提升，转化率更充分，MEK和水煮都会走高；T弯可能略微变大，体系属于更彻底的固化制度。"))
    q, a = rng.choice(rules)
    return q, a


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_all = samples_df()
    df = labeled(df_all)
    nn = build_nn(df)

    # ---------- 预训练语料 ----------
    pre = []
    for p in _tds_files():
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for i in range(0, len(t) - 60, 420):
            seg = t[i:i + 520].strip()
            if len(seg) > 80:
                pre.append({"text": seg})
    for name, m in mlib.items():
        pre.append({"text": f"{name}是项目原料库中的{m.get('role')}（{m.get('rtype')}），"
                            f"固含{m.get('NV')}%、密度{m.get('density')}、分子量{m.get('Mw')}、"
                            f"羟值{m.get('OHV')}、酸值{m.get('AV')}、Tg {m.get('Tg')}°C，"
                            f"官能团：环氧{m.get('fg_epoxy')}、羟基{m.get('fg_oh')}、羧基{m.get('fg_cooh')}。"})
    for _, r in df_all.iterrows():
        comp = {k: float(v) for k, v in r["组分"].items()}
        pre.append({"text": f"{r['体系']}体系样本{r['样本ID']}：配方{fmt_comp(comp)}，"
                            f"烘烤{r['烘烤温度']:.0f}°C×{r['烘烤时间']:.0f}min"
                            + (f"，实测{perf_str(r['T弯'], r['MEK'], r['水煮'])}。"
                               if r["标签状态"] == "实测" else "。")})
    for t in MECH_KNOWLEDGE:
        pre.append({"text": t})
    print(f"预训练: {len(pre)} 条")
    _dump("pretrain_coating.jsonl", pre)

    # ---------- SFT ----------
    sft = []

    def add(q, a, w=1):
        for _ in range(w):
            sft.append({"conversations": [{"role": "user", "content": q},
                                          {"role": "assistant", "content": a}]})

    for q, a in IDENTITY_QA:
        add(q, a, 3)
    for t in MECH_KNOWLEDGE[:8]:
        key = t.split("：")[0][:18]
        add(f"讲讲{key}方面的机理", t)
        add(f"{key.split('，')[0]}是什么原理？", t)

    tds_qa = _tds_qa()
    for q, a in tds_qa:
        add(q, a)
    print(f"  TDS QA: {len(tds_qa)}")

    # 原料档案（2种问法）
    for i, (name, m) in enumerate(mlib.items()):
        a = (f"{name}是项目原料库中的{m.get('role')}（{m.get('rtype')}），固含{m.get('NV')}%、"
             f"密度{m.get('density')}、分子量{m.get('Mw')}、羟值{m.get('OHV')}、酸值{m.get('AV')}、"
             f"Tg {m.get('Tg')}°C。")
        add(f"{name}是什么原料？有什么特性？", a)
        if i % 2 == 0:
            add(f"介绍一下原料{name}", a)

    # 实测查询（简洁）
    for _, r in df.iterrows():
        comp = {k: float(v) for k, v in r["组分"].items()}
        q = f"{r['体系']}体系样本{r['样本ID']}的实测性能怎么样？"
        a = f"样本{r['样本ID']}实测：{perf_str(r['T弯'], r['MEK'], r['水煮'])}，" \
            f"烘烤{r['烘烤温度']:.0f}°C×{r['烘烤时间']:.0f}min。"
        add(q, a)

    # 推理链：已知样本分析
    idx_all = list(df.index)
    rng.shuffle(idx_all)
    for idx in idx_all[:130]:
        r = df.loc[idx]
        same = df[df["体系"] == r["体系"]].drop(index=idx)
        nns = same.loc[nn(idx)]
        add(f"分析一下样本{r['样本ID']}，它的性能是怎么来的？",
            analysis_answer(r, nns, r["体系"]), 1)

    # 新配方推断（核心）
    made = set()
    tries = 0
    while len(made) < 240 and tries < 3000:
        tries += 1
        r = df.iloc[rng.randrange(len(df))]
        system = r["体系"]
        comp = {k: float(v) for k, v in r["组分"].items()}
        total = sum(comp.values())
        mats = sorted(comp)
        c2 = dict(comp)
        for k in rng.sample(mats, min(3, len(mats))):
            delta = rng.uniform(-0.35, 0.45) * comp[k] if comp[k] > 1 else rng.uniform(-0.5, 1.0)
            c2[k] = max(0.0, comp[k] + delta)
        c2 = {k: v for k, v in c2.items() if v > 0.3}
        t2 = sum(c2.values())
        c2 = {k: v / t2 * total for k, v in c2.items()}
        key = tuple(sorted(round(v, 1) for v in c2.values()))
        if key in made:
            continue
        made.add(key)
        bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
        if rng.random() < 0.25:
            bake = (200.0, 10.0) if bake == (205.0, 17.0) else (205.0, 17.0)
        new_mech = mech_readout(c2, mlib, bake[0], bake[1])
        same = df[df["体系"] == system]
        codes = present_codes(df)
        Xl = build_matrix(same, mlib, codes).values
        Xl = np.nan_to_num((Xl - Xl.mean(0)) / (Xl.std(0) + 1e-9))
        xn = np.nan_to_num(np.array(build_row(c2, system, bake[0], bake[1], mlib, codes)))
        d = np.linalg.norm(Xl - xn, axis=1)
        top = same.index[np.argsort(d)[:2]]
        nns = same.loc[top]
        comp_str = "、".join(f"{k} {v:.1f}%" for k, v in sorted(c2.items(), key=lambda x: -x[1]))
        q = (f"请推断这个新配方的性能：{system}体系，{comp_str}，"
             f"烘烤{bake[0]:.0f}°C×{bake[1]:.0f}min。")
        add(q, infer_answer(c2, system, bake, nns, None, new_mech), 2)

    # 反事实 + 调优
    for _ in range(90):
        r = df.iloc[rng.randrange(len(df))]
        comp = {k: float(v) for k, v in r["组分"].items()}
        q, a = counterfactual_qa(comp, r["体系"], (r["烘烤温度"], r["烘烤时间"]))
        add(f"对于{r['体系']}体系的配方，{q}，性能会怎么变化？", a, 1)
    tuning = [
        ("怎么提高涂膜的耐溶剂性（MEK）？",
         "三个抓手：一是提高固化剂占比或提高其官能度，把交联密度做上去；二是把烘烤推向205°C×17min这类更彻底的制度，提高转化率；三是保持当量比在体系经验区间内，避免固化不足。代价是T弯可能变大、涂膜变脆，柔韧与耐溶剂要取平衡。"),
        ("T弯总是偏大（偏脆），怎么改善？",
         "降低交联密度：适当减少固化剂或换用低官能度品种；选用柔性更好的环氧骨干；避免固化剂严重过量。注意柔韧性上去了，MEK擦拭和水煮通常会让出一部分，需要找到平衡点。"),
        ("水煮总是不过关（起泡/降级），怎么排查？",
         "按顺序查三件事：固化是否充分（烘烤制度和当量比）、极性小分子是否残留（溶剂/助剂比例）、涂膜是否致密（交联密度）。通常把固化做彻底、控制亲水组分比例后，水煮等级能明显改善。"),
        ("新体系没有多少历史数据，怎么办？",
         "先用机理定大方向：确定树脂与固化剂类型、按当量比设计中心点，再围绕中心点做小步正交微调；每轮只动1~2个变量，用少量实板快速迭代。等实测数据积累起来后再收敛配方窗口。"),
    ]
    for q, a in tuning:
        add(q, a, 2)

    rng.shuffle(sft)
    print(f"SFT: {len(sft)} 条")
    _dump("sft_coating.jsonl", sft)


def _tds_files():
    import glob
    return sorted(glob.glob(os.path.join(TDS_DIR, "**", "*.md"), recursive=True))


def _tds_qa():
    out = []
    for p in _tds_files():
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if "_MSDS" in os.path.basename(p) or len(t) < 120:
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        head = " ".join(t.split()).strip()
        if len(head) < 80:
            continue
        seg = head[:260]
        out.append((f"介绍一下TDS文档：{name}", f"{name}的技术资料要点：{seg}…"))
        if len(out) >= 200:
            break
    return out


def _dump(name, rows):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


import json  # noqa: E402

if __name__ == "__main__":
    main()
