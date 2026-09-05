# -*- coding: utf-8 -*-
"""推荐引擎：给定目标性能 -> 锚点检索 + 专家约束下的爬山搜索 -> 推荐配方 + 推理轨迹。"""
import copy
import random

import numpy as np

from data import ROLES, TARGETS, mat_lib, samples_df, labeled
from features import mech_readout
from knowledge import (check_formulation, rule_summary, set_material_pool,
                       ROLE_WINDOWS, BAKE_BY_SYSTEM)

DIRECTIONS = {"<=": -1, ">=": 1}


def _fmt(v, unit):
    return f"{v:.1f}{unit}" if v is not None and not np.isnan(v) else "—"


def parse_target(s):
    """'T弯<=12' / 'MEK>=100' / '水煮>=4' -> (op, value)"""
    s = s.strip().replace("≤", "<=").replace("≥", ">=")
    for op in ("<=", ">="):
        if op in s:
            k, v = s.split(op)
            return k.strip(), op, float(v)
    raise ValueError(f"目标格式错误: {s}（示例：T弯<=12 / MEK>=100 / 水煮>=4）")


def margin(tgt, pred, op, value):
    """归一化满足度：>0 表示达标，越大越好。"""
    if tgt == "水煮":
        return (pred - 0.5) / 0.5
    if op in ("<=", "≤", "<"):
        return (value - pred) / max(abs(value), 1.0)
    return (pred - value) / max(abs(value), 1.0)


class Recommender:
    def __init__(self, predictor):
        self.pred = predictor
        self.mlib = mat_lib()
        df = labeled(samples_df())
        self.lab = df
        pools = {}
        for system in df["体系"].unique():
            mats = set()
            for comp in df.loc[df["体系"] == system, "组分"]:
                mats.update(comp)
            pools[system] = mats
        self.pools = pools
        self.pool_roles = {s: {r: sorted(m for m in ms if self.mlib[m].get("role") == r)
                               for r in ROLES} for s, ms in pools.items()}
        set_material_pool(pools)

    # ---------------- 评分 ----------------
    def evaluate(self, comp, system, bake_temp, bake_time, targets, mech=None):
        p = self.pred.predict(comp, system, bake_temp, bake_time)
        scores = {}
        for tgt, (op, value) in targets.items():
            key = "水煮≥4概率" if tgt == "水煮" else tgt
            scores[tgt] = margin(tgt, p[key], op, value)
        if mech is None:
            mech = mech_readout(comp, self.mlib, bake_temp, bake_time)
        results = check_formulation(comp, self.mlib, system, bake_temp, bake_time, mech)
        errs, warns, oks = rule_summary(results)
        hard = sum(1 for lv, _ in results if lv == "error")
        soft = sum(1 for lv, _ in results if lv == "warn")
        base = float(np.mean(list(scores.values())))
        score = base - 0.6 * hard - 0.08 * soft
        return dict(pred=p, scores=scores, score=score, hard=hard, soft=soft,
                    rules=results, mech=mech)

    # ---------------- 扰动算子 ----------------
    def _renorm(self, comp, total):
        s = sum(comp.values())
        f = total / s if s > 0 else 1.0
        return {k: round(v * f, 4) for k, v in comp.items()}

    def mutate(self, comp, system, rng):
        c = dict(comp)
        pool = self.pool_roles.get(system, {})
        op = rng.random()
        mats = sorted(c)
        if op < 0.45 and mats:
            k = rng.choice(mats)
            delta = rng.choice([-3, -2, -1, -0.5, 0.5, 1, 2, 3])
            c[k] = max(0.0, c[k] + delta)
            if c[k] < 0.05:
                del c[k]
        elif op < 0.75 and mats:
            k = rng.choice(mats)
            role = self.mlib[k].get("role")
            cands = pool.get(role, [])
            cands = [m for m in cands if m != k]
            if cands and c[k] > 0.05:
                m2 = rng.choice(cands)
                c[m2] = c.get(m2, 0) + c[k]
                del c[k]
        elif op < 0.92:
            res = [k for k in mats if self.mlib[k].get("role") == "树脂"]
            cures = [k for k in mats if self.mlib[k].get("role") == "固化剂"]
            cur = sum(c[k] for k in res)
            cure_sum = sum(c[k] for k in cures)
            if res and cures and cur > 5:
                ratio = rng.uniform(0.08, 0.45)
                new_cure = cur * ratio
                for k in cures:
                    c[k] = c[k] * (new_cure / max(cure_sum, 0.1))
        else:
            k = rng.choice(sorted(pool.get(rng.choice(ROLES), []) or [""]))
            if k and k not in c:
                c[k] = round(rng.uniform(0.2, 1.5), 3)
        return self._renorm(c, 100.0)

    # ---------------- 主入口 ----------------
    def recommend(self, system, targets, top_n=5, bake=None, n_iter=3000, seed=42,
                  verbose=True):
        rng = random.Random(seed)
        tdict = {}
        for tgt in targets:
            k, op, v = parse_target(tgt)
            tdict[k] = (op, v)

        pool = self.pools.get(system)
        if not pool:
            raise ValueError(f"体系 {system} 无实测数据")
        lab_s = self.lab[self.lab["体系"] == system]
        if bake:
            bt, btm = bake
            lab_b = lab_s[(lab_s["烘烤温度"] == bt) & (lab_s["烘烤时间"] == btm)]
            if len(lab_b) >= 10:
                lab_s = lab_b
        if bake:
            bake_temp, bake_time = bake
        else:
            bake_temp, bake_time = BAKE_BY_SYSTEM[system][0]

        # 锚点：按目标满足度排序的实测配方
        anchor_scores = []
        for _, r in lab_s.iterrows():
            sc = []
            for tgt, (op, v) in tdict.items():
                y = r[tgt]
                if tgt == "水煮":
                    sc.append(margin(tgt, 1.0 if y >= 4 else 0.0, op, v))
                elif not np.isnan(y):
                    sc.append(margin(tgt, y, op, v))
            if sc:
                anchor_scores.append((float(np.mean(sc)), r))
        anchor_scores.sort(key=lambda x: -x[0])
        anchors = [r for _, r in anchor_scores[:12]]
        if not anchors:
            anchors = [r for _, r in lab_s.iterrows()]

        cand = {}
        for a in anchors:
            comp0 = {k: float(v) for k, v in a["组分"].items()}
            t0 = sum(comp0.values())
            comp0 = {k: v / t0 * 100.0 for k, v in comp0.items()}
            c = copy.deepcopy(comp0)
            ev = self.evaluate(c, system, bake_temp, bake_time, tdict)
            src = (f"锚点样本 {a['样本ID']}（实测 T弯={_fmt(a['T弯'], 'mm')}, "
                   f"MEK={_fmt(a['MEK'], '次')}, 水煮={_fmt(a['水煮'], '级')}）")
            cand[tuple(sorted(c.items()))] = (ev, c, src + "，未改动")
            cur = ev["score"]
            for _ in range(n_iter // len(anchors)):
                c2 = self.mutate(c, system, rng)
                ev2 = self.evaluate(c2, system, bake_temp, bake_time, tdict)
                if ev2["score"] >= cur - 0.01:
                    if ev2["score"] > cur or (ev2["score"] == cur and rng.random() < 0.5):
                        key = tuple(sorted(c2.items()))
                        if key not in cand:
                            cand[key] = (ev2, c2, f"锚点 {a['样本ID']} 起步，爬山搜索改进")
                        c, cur, ev = c2, ev2["score"], ev2
        ranked = sorted(cand.values(), key=lambda x: -x[0]["score"])
        # 多样性：成分向量余弦去重
        picked, used = [], []
        for ev, comp, src in ranked:
            v = np.array([comp.get(m, 0.0) for m in sorted(self.pools[system])])
            if any(np.linalg.norm(v - u) / (np.linalg.norm(u) + 1e-9) < 0.08 for u in used):
                continue
            picked.append((ev, comp, src))
            used.append(v)
            if len(picked) >= top_n:
                break
        if verbose:
            self._report(system, bake_temp, bake_time, tdict, picked)
        return picked

    def _report(self, system, bt, btm, tdict, picked):
        print(f"\n== {system} 体系推荐（目标："
              + "；".join(f"{k} {op} {v}" for k, (op, v) in tdict.items())
              + f"，烘烤 {bt}°C×{btm}min）==")
        for i, (ev, comp, src) in enumerate(picked, 1):
            pred = ev["pred"]
            perf = "、".join(
                f"{k}{'≥' if tdict[k][0] == '>=' else '≤'}{tdict[k][1]}→预测"
                + ("P=%.0f%%" % (pred["水煮≥4概率"] * 100) if k == "水煮" else f"{pred[k]:.1f}")
                + (f"(满足)" if ev["scores"][k] > 0 else "(未满足)")
                for k in tdict)
            print(f"\n推荐 {i} | 综合分 {ev['score']:.3f} | {perf}")
            print("  配方: " + "、".join(f"{k}({self.mlib[k].get('role')}){v:.2f}%"
                                       for k, v in sorted(comp.items(), key=lambda x: -x[1])))
            errs, warns, oks = rule_summary(ev["rules"])
            print(f"  专家规则: {len(oks)}通过 / {len(warns)}提醒 / {len(errs)}违规"
                  + ("" if not warns else "；" + "；".join(warns[:2])))
            print(f"  来源: {src}")
