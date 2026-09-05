# -*- coding: utf-8 -*-
"""不同容忍度下的预测命中率评估（5折OOF，诚实口径）。"""
import sys
import os

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import mat_lib, samples_df, labeled  # noqa: E402
from features import build_matrix  # noqa: E402
from predictor import (CoatingPredictor, transform, restore, series_enc,  # noqa: E402
                       _fit_reg, _fit_clf, _blend, _t弯_filter_inner, _p_hi_oof, MEK_CAP)


def oof_target(tgt, X, series, df):
    m = df[tgt].notna().values
    y = df.loc[m, tgt].values.astype(float)
    y_t = transform(tgt, y)
    Xm = X[m]
    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=42).split(Xm):
        se_tr, se_te, _, _ = series_enc(series[m][tr], y_t[tr], series[m][te])
        Xtr = np.column_stack([Xm[tr], se_tr])
        Xte = np.column_stack([Xm[te], se_te])
        if tgt == "T弯":
            keep = _t弯_filter_inner(Xtr, y_t[tr], series[m][tr])
            Xtr2, ytr = Xtr[keep], y_t[tr][keep]
        else:
            yb_tr = (y[tr] >= MEK_CAP).astype(int)
            c = _fit_clf(Xtr, yb_tr, "xgb")
            Xte = np.column_stack([Xte, c.predict_proba(Xte)[:, 1]])
            p_hi_tr = _p_hi_oof(Xtr, y[tr], series[m][tr])
            Xtr2 = np.column_stack([Xtr, p_hi_tr])
            ytr = y_t[tr]
        pv = _blend(_fit_reg(Xtr2, ytr, "xgb").predict(Xte),
                    _fit_reg(Xtr2, ytr, "lgb").predict(Xte))
        oof[te] = restore(tgt, pv)
    return y, oof


def main():
    p = CoatingPredictor.load()
    df = labeled(samples_df())
    mlib = mat_lib()
    X = build_matrix(df, mlib, p.codes).values
    series = df["系列"].values

    for tgt, tols, unit in [("T弯", [1, 2, 3, 5], "mm"), ("MEK", [15, 30, 50, 80], "次")]:
        y, pv = oof_target(tgt, X, series, df)
        print(f"{tgt} (n={len(y)}, 全体标准差σ={np.std(y):.1f}{unit}):")
        for tol in tols:
            print(f"  绝对误差≤{tol}{unit}: 命中率 {np.mean(np.abs(y - pv) <= tol) * 100:.1f}%")
        print(f"  相对误差≤20%: {np.mean(np.abs(y - pv) <= 0.2 * np.maximum(y, 1)) * 100:.1f}%"
              f"   ≤30%: {np.mean(np.abs(y - pv) <= 0.3 * np.maximum(y, 1)) * 100:.1f}%")

    m = df["水煮"].notna().values
    yb = (df.loc[m, "水煮"] >= 4).astype(int).values
    Xm = X[m]
    oofp = np.zeros(len(yb))
    for tr, te in KFold(5, shuffle=True, random_state=42).split(Xm):
        se_tr, se_te, _, _ = series_enc(series[m][tr], df.loc[m, "水煮"].values[tr], series[m][te])
        Xtr = np.column_stack([Xm[tr], se_tr])
        Xte = np.column_stack([Xm[te], se_te])
        px = _fit_clf(Xtr, yb[tr], "xgb").predict_proba(Xte)[:, 1]
        pl = _fit_clf(Xtr, yb[tr], "lgb").predict_proba(Xte)[:, 1]
        oofp[te] = 0.85 * px + 0.15 * pl
    print(f"水煮≥4 判级 (n={len(yb)}, 正例占比 {yb.mean() * 100:.0f}%):")
    print(f"  阈值0.5准确率: {np.mean((oofp >= 0.5) == yb) * 100:.1f}%")
    acc_opt = max(np.mean((oofp >= t) == yb) for t in np.linspace(0.3, 0.7, 9))
    print(f"  调优阈值最高: {acc_opt * 100:.1f}%")
    hi = np.abs(oofp - 0.5) > 0.25
    if hi.sum() > 10:
        print(f"  高置信子集(占{hi.mean() * 100:.0f}%): 准确率 {np.mean((oofp[hi] >= 0.5) == yb[hi]) * 100:.1f}%")
    prec = np.mean(yb[oofp >= 0.6]) if (oofp >= 0.6).sum() > 5 else None
    if prec is not None:
        print(f"  预测≥4且概率≥0.6时: 精确率 {prec * 100:.1f}%（报'达标'的可信度）")


if __name__ == "__main__":
    main()
