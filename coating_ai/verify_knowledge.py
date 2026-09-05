# -*- coding: utf-8 -*-
"""真知识检验套件：标签打乱对照 + 配方族留出 + 机理特征消融 + 方向一致性。"""
import os
import sys

import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import mat_lib, samples_df, labeled  # noqa: E402
from features import build_matrix  # noqa: E402
from predictor import (transform, restore, series_enc, _fit_reg, _fit_clf,  # noqa: E402
                       _blend, _t弯_filter_inner, _p_hi_oof, MEK_CAP)

SEED = 42


def cv_eval(X, y, series, tgt, groups=None, n_splits=5, shuffle_labels=False, seed=SEED):
    """返回OOF预测。shuffle_labels=True 时打乱目标（对照组）。"""
    rng = np.random.RandomState(seed)
    if shuffle_labels:
        y = y[rng.permutation(len(y))]
    y_t = transform(tgt, y)
    oof = np.zeros(len(y))
    if groups is not None:
        splits = list(GroupKFold(n_splits).split(X, groups=groups))
    else:
        splits = list(KFold(n_splits, shuffle=True, random_state=seed).split(X))
    for tr, te in splits:
        se_tr, se_te, _, _ = series_enc(series[tr], y_t[tr], series[te])
        Xtr = np.column_stack([X[tr], se_tr])
        Xte = np.column_stack([X[te], se_te])
        if tgt == "T弯":
            keep = _t弯_filter_inner(Xtr, y_t[tr], series[tr])
            Xtr2, ytr = Xtr[keep], y_t[tr][keep]
        else:
            yb_tr = (y[tr] >= MEK_CAP).astype(int)
            c = _fit_clf(Xtr, yb_tr, "xgb")
            Xte = np.column_stack([Xte, c.predict_proba(Xte)[:, 1]])
            p_hi = _p_hi_oof(Xtr, y[tr], series[tr])
            Xtr2 = np.column_stack([Xtr, p_hi])
            ytr = y_t[tr]
        pv = _blend(_fit_reg(Xtr2, ytr, "xgb").predict(Xte),
                    _fit_reg(Xtr2, ytr, "lgb").predict(Xte))
        oof[te] = restore(tgt, pv)
    return y, oof


def main():
    df = labeled(samples_df())
    mlib = mat_lib()
    from data import present_codes
    X = build_matrix(df, mlib, present_codes(df)).values
    series = df["系列"].values
    groups = df["系列"].values

    print("=" * 60)
    print("实验1：标签打乱对照（真信号 vs 过拟合假象的金标准）")
    print("  若打乱标签后模型'准确率'不塌方，说明学到的是数据泄漏/假象")
    for tgt, tol, unit in [("T弯", 3, "mm"), ("MEK", 50, "次")]:
        m = df[tgt].notna().values
        y_real, pv_real = cv_eval(X[m], df.loc[m, tgt].values.astype(float),
                                  series[m], tgt, shuffle_labels=False)
        mae_real = mean_absolute_error(y_real, pv_real)
        y_shuf, pv_shuf = cv_eval(X[m], df.loc[m, tgt].values.astype(float),
                                  series[m], tgt, shuffle_labels=True)
        mae_shuf = mean_absolute_error(y_shuf, pv_shuf)
        base = np.mean(np.abs(y_real - np.median(y_real)))
        print(f"  {tgt}: 真实MAE={mae_real:.2f}{unit} | 打乱标签MAE={mae_shuf:.2f}{unit} "
              f"| 朴素基线(中位数)={base:.2f}{unit}")
        print(f"    打乱后命中≤{tol}{unit}: {np.mean(np.abs(y_shuf - pv_shuf) <= tol) * 100:.1f}%"
              f"（真实: {np.mean(np.abs(y_real - pv_real) <= tol) * 100:.1f}%）")

    print("=" * 60)
    print("实验2：配方族整族留出（整个系列从未见过，比随机5折更严格）")
    for tgt, tol, unit in [("T弯", 3, "mm"), ("MEK", 50, "次")]:
        m = df[tgt].notna().values
        y_r, pv_r = cv_eval(X[m], df.loc[m, tgt].values.astype(float), series[m], tgt)
        y_g, pv_g = cv_eval(X[m], df.loc[m, tgt].values.astype(float), series[m], tgt,
                            groups=groups[m], n_splits=5)
        print(f"  {tgt}: 随机5折 MAE={mean_absolute_error(y_r, pv_r):.2f}{unit}"
              f" | 整族留出 MAE={mean_absolute_error(y_g, pv_g):.2f}{unit}")
        print(f"    整族留出 命中≤{tol}{unit}: {np.mean(np.abs(y_g - pv_g) <= tol) * 100:.1f}%"
              f"（随机: {np.mean(np.abs(y_r - pv_r) <= tol) * 100:.1f}%）")
        r2_g = r2_score(y_g, pv_g)
        print(f"    整族留出 R²={r2_g:.3f}")

    print("=" * 60)
    print("实验3：机理特征消融（知识载体归因）")
    n_codes = sum(1 for _ in p_codes(df)) if False else None
    from data import present_codes
    codes = present_codes(df)
    n_comp = len(codes)
    n_role = 5
    n_mech = X.shape[1] - n_comp - n_role - 3 - 3
    blocks = {"仅组分用量": list(range(n_comp)),
              "仅机理特征": list(range(n_comp + n_role, n_comp + n_role + n_mech)),
              "全特征": list(range(X.shape[1]))}
    for name, cols in blocks.items():
        m = df["T弯"].notna().values
        Xb = X[m][:, cols]
        oof = np.zeros(m.sum())
        y = df.loc[m, "T弯"].values.astype(float)
        y_t = np.sqrt(np.clip(y, 0, None))
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xb):
            se_tr, se_te, _, _ = series_enc(series[m][tr], y_t[tr], series[m][te])
            Xtr = np.column_stack([Xb[tr], se_tr])
            Xte = np.column_stack([Xb[te], se_te])
            keep = _t弯_filter_inner(Xtr, y_t[tr], series[m][tr])
            mdl = _fit_reg(Xtr[keep], y_t[tr][keep], "xgb")
            oof[te] = np.clip(mdl.predict(Xte), 0, None) ** 2
        print(f"  T弯 {name:8s}: MAE={mean_absolute_error(y, oof):.2f}mm  R²={r2_score(y, oof):.3f}")

    print("=" * 60)
    print("实验4：因果方向一致性（GBM 侧）——提高固化剂50%，机理上 MEK应升/T弯应升")
    rng = np.random.RandomState(7)
    from predictor import CoatingPredictor
    pred = CoatingPredictor.load()
    consistent_m, consistent_t, n_pairs = 0, 0, 0
    for _ in range(40):
        r = df.iloc[rng.randint(len(df))]
        comp = {k: float(v) for k, v in r["组分"].items()}
        cures = [k for k in comp if mlib[k].get("role") == "固化剂"]
        if not cures:
            continue
        total = sum(comp.values())
        c2 = dict(comp)
        for k in cures:
            c2[k] = comp[k] * 1.5
        f = total / sum(c2.values())
        c2 = {k: v * f for k, v in c2.items()}
        base = pred.predict(comp, r["体系"], float(r["烘烤温度"]), float(r["烘烤时间"]))
        var = pred.predict(c2, r["体系"], float(r["烘烤温度"]), float(r["烘烤时间"]))
        n_pairs += 1
        consistent_m += var["MEK"] > base["MEK"]
        consistent_t += var["T弯"] > base["T弯"]
    print(f"  {n_pairs} 对样本：MEK 方向正确率 {consistent_m}/{n_pairs}"
          f"，T弯方向正确率 {consistent_t}/{n_pairs}")


def p_codes(df):
    from data import present_codes
    return present_codes(df)


if __name__ == "__main__":
    main()
