# -*- coding: utf-8 -*-
"""数据加载：merged_data.pkl -> 样本表 + 材料库。"""
import os
import pickle

import numpy as np
import pandas as pd

PKL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "3NEW", "generalization", "data", "merged_data.pkl")

ROLES = ["树脂", "固化剂", "溶剂", "助剂", "颜料"]
TARGETS = ["T弯", "MEK", "水煮"]
SYSTEMS = ["环氧酚醛", "环氧配比方案", "聚酯金黄"]


def load_raw():
    with open(PKL, "rb") as f:
        return pickle.load(f)


def mat_lib():
    return load_raw()["full_mat"]


def samples_df():
    d = load_raw()
    rows = []
    for s in d["all_samples"]:
        comp = {k: float(v) for k, v in s["组分"].items()}
        rows.append({
            "样本ID": s["样本ID"], "体系": s["体系"], "系列": s["系列"],
            "组分": comp, "组分总计": float(sum(comp.values())),
            "烘烤温度": s.get("烘烤温度"), "烘烤时间": s.get("烘烤时间"),
            "T弯": s.get("T弯"), "MEK": s.get("MEK"), "水煮": s.get("水煮"),
            "标签状态": s.get("标签状态"),
        })
    df = pd.DataFrame(rows)
    for t in TARGETS:
        df[t] = pd.to_numeric(df[t], errors="coerce")
    return df


def present_codes(df=None):
    if df is None:
        df = samples_df()
    codes = set()
    for comp in df["组分"]:
        codes.update(comp)
    return sorted(codes)


def labeled(df):
    return df[df["标签状态"] == "实测"].reset_index(drop=True)
