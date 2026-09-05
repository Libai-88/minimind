# -*- coding: utf-8 -*-
"""补充精修数据：反事实泛化问法 + 极端配方坦诚回答 + 推断问法改写。"""
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "coating_ai"))

from data import mat_lib, samples_df, labeled  # noqa: E402
from features import mech_readout  # noqa: E402

mlib = mat_lib()
rng = random.Random(7788)

rows = []


def add(q, a, w=1):
    for _ in range(w):
        rows.append({"conversations": [{"role": "user", "content": q},
                                       {"role": "assistant", "content": a}]})


COUNTERFACTUAL_GENERIC = [
    ("把固化剂比例提高约50%",
     "交联密度上升，MEK擦拭次数会明显提高，水煮等级通常随之上升；代价是涂膜变硬变脆，T弯数值变大，柔韧性下降。"),
    ("把固化剂比例降到约一半",
     "交联不足，MEK擦拭次数会大幅下降，可能掉到几十次的量级，水煮等级也难以维持；好处是涂膜更柔韧，T弯变小。"),
    ("把溶剂比例提高约三分之一",
     "固含下降，施工粘度更低、流平更好，但挥发物增多，涂膜致密度略降，耐溶剂与水煮的上限会受影响，烘烤负荷也更大。"),
    ("把蜡粉比例提高一倍",
     "滑爽性和抗划伤会改善，但过量蜡会降低层间附着并在烘烤中上浮产生雾影，需要控制在较小的比例窗口内。"),
    ("把烘烤从200°C×10min改成205°C×17min",
     "等效固化强度明显提升，转化率更充分，MEK和水煮都会走高；T弯可能略微变大。"),
    ("把助剂（流平剂、蜡粉）总量减少一半",
     "涂膜表面状态可能变差，出现橘皮或划痕敏感；但对交联与耐性影响不大，属于外观层面的取舍。"),
]
Q_TEMPLATES = [
    "对于{system}体系的配方，{change}，性能会怎么变化？",
    "{system}体系里如果{change}，各性能指标大致会怎么走？",
    "在{system}配方基础上，{change}，请推断性能变化。",
    "如果把一个{system}配方中的{change}，会发生什么？",
]

systems = ["环氧酚醛", "环氧配比方案", "聚酯金黄"]
for _ in range(70):
    q_t = rng.choice(Q_TEMPLATES)
    change, a = rng.choice(COUNTERFACTUAL_GENERIC)
    s = rng.choice(systems)
    add(q_t.format(system=s, change=change), a)

TUNING_GENERIC = [
    ("怎么提高涂膜的耐溶剂性（MEK擦拭次数）？",
     "三个抓手：提高固化剂占比或官能度把交联密度做上去；把烘烤推向205°C×17min这类更彻底的制度；保持当量比在体系经验区间。代价是T弯变大、涂膜变脆。"),
    ("T弯偏大（偏脆）怎么改善？",
     "降低交联密度：减少固化剂或换低官能度品种、选用柔性环氧骨干、避免固化剂过量。柔韧性上去了，MEK和水煮通常要让出一部分。"),
    ("水煮不过关怎么排查？",
     "按顺序查固化是否充分（烘烤制度和当量比）、极性小分子是否残留（溶剂助剂比例）、涂膜是否致密（交联密度）。"),
]
for _ in range(25):
    q, a = rng.choice(TUNING_GENERIC)
    v = rng.choice([q, q.replace("？", "？请给机理依据。"), "配方专家，" + q])
    add(v, a)

# ---------- 极端配方：坦诚回答 ----------
df = labeled(samples_df())


def infer_style():
    return ("我按机理来分析这个配方：这个组成明显偏离了项目历史数据范围，"
            "我只能给方向性判断：{point}。"
            "建议把它当作探索性试验，小样试板验证后再放大，不要直接量产。")


OOD_CASES = []
# 溶剂占绝对主导
for _ in range(12):
    r = df.iloc[rng.randrange(len(df))]
    comp = {k: float(v) for k, v in r["组分"].items() if mlib[k].get("role") == "树脂"}
    comp[rng.choice([k for k, v in mlib.items() if v.get("role") == "溶剂"])] = 55.0
    comp[rng.choice([k for k, v in mlib.items() if v.get("role") == "固化剂"])] = 8.0
    OOD_CASES.append((comp, r["体系"],
                      "溶剂占到一半以上，固含极低，成膜会严重偏薄甚至不连续；"
                      "固化剂虽然比例看似正常，但基料太少，交联网络无法有效搭接，"
                      "MEK擦拭会非常低，水煮基本不合格。"))
# 固化剂绝对主导
for _ in range(12):
    r = df.iloc[rng.randrange(len(df))]
    comp = {k: float(v) for k, v in r["组分"].items() if mlib[k].get("role") == "树脂"}
    comp[rng.choice([k for k, v in mlib.items() if v.get("role") == "溶剂"])] = 10.0
    comp[rng.choice([k for k, v in mlib.items() if v.get("role") == "固化剂"])] = 45.0
    OOD_CASES.append((comp, r["体系"],
                      "固化剂占近一半，远超环氧树脂的当量需求，大量残留的小分子固化剂会使涂膜发脆、"
                      "可能渗出发雾；交联点虽多但网络不规整，T弯会明显变差，耐性提升也有限。"))
# 单一树脂极简
for _ in range(10):
    r = df.iloc[rng.randrange(len(df))]
    resins = [k for k, v in mlib.items() if v.get("role") == "树脂"]
    comp = {rng.choice(resins): 85.0, rng.choice([k for k, v in mlib.items() if v.get("role") == "溶剂"]): 12.0,
            rng.choice([k for k, v in mlib.items() if v.get("role") == "固化剂"]): 3.0}
    OOD_CASES.append((comp, r["体系"],
                      "这是只有一种树脂加固化剂的极简配方，缺少流平与抗性助剂，外观上易出现橘皮缩孔；"
                      "固化剂仅3%明显不足，交联不充分，MEK擦拭偏低，T弯会不错但耐性上不去。"))

for comp, system, point in OOD_CASES:
    comp_str = "、".join(f"{k} {v:.1f}%" for k, v in sorted(comp.items(), key=lambda x: -x[1]))
    bake = rng.choice([(200.0, 10.0), (205.0, 17.0)])
    q = f"请推断这个新配方的性能：{system}体系，{comp_str}，烘烤{bake[0]:.0f}°C×{bake[1]:.0f}min。"
    qv = rng.choice([q, q.replace("请推断这个新配方的性能", "帮我看看这个配方性能如何"),
                     q.replace("请推断这个新配方的性能", "预测一下这个配方")])
    add(qv, infer_style().format(point=point), 2)

# 常规配方的问法改写（不改变答案，只换问法）
for _ in range(45):
    r = df.iloc[rng.randrange(len(df))]
    comp = {k: float(v) for k, v in r["组分"].items()}
    total = sum(comp.values())
    comp = {k: round(v / total * 100, 1) for k, v in comp.items()}
    comp_str = "、".join(f"{k} {v}%" for k, v in sorted(comp.items(), key=lambda x: -x[1]))
    system = r["体系"]
    bake = (float(r["烘烤温度"]), float(r["烘烤时间"]))
    mech = mech_readout(comp, mlib, bake[0], bake[1])
    base = (f"组成：树脂约{sum(v for k, v in comp.items() if mlib[k].get('role') == '树脂'):.0f}%，"
            f"固化剂约{sum(v for k, v in comp.items() if mlib[k].get('role') == '固化剂'):.0f}%，"
            f"溶剂约{sum(v for k, v in comp.items() if mlib[k].get('role') == '溶剂'):.0f}%；"
            f"当量比落在体系常规区间，配方处于历史数据覆盖范围内。")
    perf = (f"参照历史相近配方，推断T弯约{max(r['T弯'], 0) * 0.85:.0f}~{r['T弯'] * 1.2:.0f}mm，"
            f"MEK擦拭约{max(r['MEK'], 5) * 0.7:.0f}~{max(r['MEK'], 5) * 1.35:.0f}次，"
            f"水煮大概率{max(r['水煮'], 3):.0f}级；"
            "区间把握度中等，建议按标准工艺试板验证。")
    for q_t in ["帮我看看这个配方性能如何：{s}体系，{c}，烘烤{t:.0f}°C×{m:.0f}min。",
                "预测一下这个配方：{s}体系，{c}，烘烤{t:.0f}°C×{m:.0f}min。",
                "请推断这个新配方的性能：{s}体系，{c}，烘烤{t:.0f}°C×{m:.0f}min。"]:
        add(q_t.format(s=system, c=comp_str, t=bake[0], m=bake[1]),
            f"我按机理来分析：{base}{perf}", 1)

rng.shuffle(rows)
out = os.path.join(HERE, "dataset", "sft_refine.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"精修数据 {len(rows)} 条 -> {out}")
