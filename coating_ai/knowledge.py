# -*- coding: utf-8 -*-
"""专家知识规则库：配比窗口、化学计量比、工艺窗口、原料相容性。

窗口数值 = 领域化学先验 + 从 3NEW 实测数据（421 样本）标定的体系指纹。
规则输出 (规则名, 级别[error/warn/ok], 说明) 供预测与推荐共用。
"""
import numpy as np

from data import ROLES

# 体系 -> 角色质量占比窗口 (%)：[min, max]，取体系实测 mean±2.5σ 并夹化学合理区间
ROLE_WINDOWS = {
    "环氧酚醛": {
        "树脂": (50, 88), "固化剂": (3, 28), "溶剂": (2, 24),
        "助剂": (0.2, 8), "颜料": (0, 2),
    },
    "环氧配比方案": {
        "树脂": (38, 88), "固化剂": (2, 16), "溶剂": (6, 20),
        "助剂": (0.3, 10), "颜料": (0, 35),
    },
    "聚酯金黄": {
        "树脂": (78, 97), "固化剂": (1.5, 16), "溶剂": (0.5, 7),
        "助剂": (0, 1.5), "颜料": (0, 2),
    },
}

# 化学计量比窗口（体系指纹 ± 容差）
STOICH_WINDOWS = {
    "环氧酚醛": {"r_phenol_epoxy": (0.03, 0.32)},
    "环氧配比方案": {"r_phenol_epoxy": (0.5, 1.5), "r_nco_oh": (0.0, 0.06)},
    "聚酯金黄": {"r_nco_oh": (0.08, 0.22), "r_phenol_epoxy": (0.0, 1.0)},
}

# 工艺窗口 (温度°C, 时间min)：项目实际只用两种烘烤制度
BAKE_REGIMES = [(200, 10), (205, 17)]
BAKE_BY_SYSTEM = {
    "环氧酚醛": [(200, 10), (205, 17)],
    "环氧配比方案": [(205, 17)],
    "聚酯金黄": [(205, 17)],
}

# 体系在 3NEW 实测数据中登记过的原料（推荐搜索的原料池，保证相容性）
MATERIAL_POOL = {}

MATERIAL_POOL_NOTES = {
    "环氧酚醛": "环氧树脂(IR/…) + 酚醛固化剂(RF…) + 混合溶剂 + 流平/蜡类助剂",
    "环氧配比方案": "环氧体系 + 色浆(颜料) + 胺类促进剂",
    "聚酯金黄": "聚酯树脂 + 封端异氰酸酯固化剂(RY…) + 少量助剂",
}


def set_material_pool(system_pools):
    MATERIAL_POOL.clear()
    MATERIAL_POOL.update(system_pools)


def check_role_windows(comp, mlib, system):
    out = []
    total = sum(comp.values()) or 1.0
    role_sum = {r: 0.0 for r in ROLES}
    for k, w in comp.items():
        role = mlib[k].get("role")
        if role in role_sum:
            role_sum[role] += w
    for r in ROLES:
        pct = role_sum[r] / total * 100
        win = ROLE_WINDOWS[system][r]
        if r == "颜料" and win[1] <= 2 and pct <= win[1] + 3:
            continue
        if pct < win[0] - 2 or pct > win[1] + 2:
            out.append(("error",
                        f"{r}占比 {pct:.1f}% 超出{system}体系窗口 [{win[0]}, {win[1]}]%"))
        elif pct < win[0] or pct > win[1]:
            out.append(("warn", f"{r}占比 {pct:.1f}% 接近{system}体系窗口边界 [{win[0]}, {win[1]}]%"))
        else:
            out.append(("ok", f"{r}占比 {pct:.1f}% 在窗口 [{win[0]}, {win[1]}]% 内"))
    return out


def check_stoich(mech, system):
    out = []
    for name, win in STOICH_WINDOWS[system].items():
        v = mech.get(name)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        lo, hi = win
        if v < lo * 0.8 or v > hi * 1.2:
            out.append(("error", f"{name}={v:.3f} 超出合理化学计量区间 [{lo:.2f}, {hi:.2f}]"))
        elif v < lo or v > hi:
            out.append(("warn", f"{name}={v:.3f} 偏离区间 [{lo:.2f}, {hi:.2f}]"))
        else:
            out.append(("ok", f"{name}={v:.3f} 计量比正常"))
    return out


def check_bake(bake_temp, bake_time, system):
    regimes = BAKE_BY_SYSTEM[system]
    if (bake_temp, bake_time) in regimes:
        return [("ok", f"烘烤 {bake_temp}°C×{bake_time}min 为{system}体系标准制度")]
    if bake_temp is None or bake_time is None:
        return [("warn", "未记录烘烤条件，固化相关预测不确定性增大")]
    return [("error",
             f"烘烤 {bake_temp}°C×{bake_time}min 非标准制度，可选："
             + " 或 ".join(f"{t}°C×{m}min" for t, m in regimes))]


def check_pool(comp, system):
    out = []
    pool = MATERIAL_POOL.get(system, set())
    if not pool:
        return out
    for k in comp:
        if k not in pool:
            out.append(("warn", f"原料 {k} 未在{system}体系实测记录中使用过（相容性未经项目验证）"))
    return out


def check_total(comp):
    out = []
    total = sum(comp.values())
    if not (85 <= total <= 115):
        out.append(("error", f"配方总量 {total:.1f}% 偏离 100%，请检查单位"))
    else:
        out.append(("ok", f"配方总量 {total:.1f}%"))
    return out


def check_formulation(comp, mlib, system, bake_temp, bake_time, mech=None):
    results = []
    results += check_total(comp)
    results += check_role_windows(comp, mlib, system)
    if mech is not None:
        results += check_stoich(mech, system)
    results += check_bake(bake_temp, bake_time, system)
    results += check_pool(comp, system)
    return results


def rule_summary(results):
    errs = [m for lv, m in results if lv == "error"]
    warns = [m for lv, m in results if lv == "warn"]
    oks = [m for lv, m in results if lv == "ok"]
    return errs, warns, oks
