# -*- coding: utf-8 -*-
"""特征工程：组分用量 + 角色占比 + 配方级机理特征 + 工艺 + 体系。"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "3NEW", "generalization", "workbench"))

from mech_desc import mech_vector, mech_features, MECH_FEATURES  # noqa: E402
from data import ROLES, SYSTEMS  # noqa: E402

SYSTEM_COL = {s: f"体系_{s}" for s in SYSTEMS}


def feature_names(codes):
    return (codes + [f"角色占比_{r}" for r in ROLES] + list(MECH_FEATURES)
            + ["烘烤温度", "烘烤时间", "烘烤强度"] + list(SYSTEM_COL.values()))


def build_row(comp, system, bake_temp, bake_time, mlib, codes):
    comp = {k: float(v) for k, v in comp.items()}
    unknown = [k for k in comp if k not in mlib]
    if unknown:
        raise KeyError(f"未知原料: {unknown}")
    total = sum(comp.values())
    if total <= 0:
        raise ValueError("配方组分总和必须大于 0")
    comp = {k: v / total * 100.0 for k, v in comp.items()}
    feats = [comp.get(c, 0.0) for c in codes]

    role_sum = {r: 0.0 for r in ROLES}
    for k, w in comp.items():
        role = mlib[k].get("role")
        if role in role_sum:
            role_sum[role] += w
    feats.extend(role_sum[r] for r in ROLES)

    feats.extend(mech_vector(comp, mlib, bake_temp, bake_time))

    bt = float(bake_temp) if bake_temp is not None and not np.isnan(bake_temp) else np.nan
    btm = float(bake_time) if bake_time is not None and not np.isnan(bake_time) else np.nan
    feats.extend([bt, btm, bt * btm / 1000.0 if bt == bt and btm == btm else np.nan])

    feats.extend(1.0 if system == s else 0.0 for s in SYSTEMS)
    return feats


def build_matrix(df, mlib, codes):
    rows = [build_row(r["组分"], r["体系"], r["烘烤温度"], r["烘烤时间"], mlib, codes)
            for _, r in df.iterrows()]
    return pd.DataFrame(rows, columns=feature_names(codes), index=df.index)


def mech_readout(comp, mlib, bake_temp, bake_time):
    comp = {k: float(v) for k, v in comp.items()}
    total = sum(comp.values())
    if total > 0:
        comp = {k: v / total * 100.0 for k, v in comp.items()}
    d, err = mech_features(comp, mlib, bake_temp, bake_time)
    return d if d is not None else {}
