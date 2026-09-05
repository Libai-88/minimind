# -*- coding: utf-8 -*-
"""预测引擎：三目标模型（T弯/MEK/水煮）。

协议要点（与 3NEW 项目 mvp81~mvp87 实验验证一致）：
  · 系列目标编码：折叠内 OOF 平滑编码（防泄漏），新配方未知系列 → 全局均值
  · T弯：sqrt 变换 + OOF 噪声过滤（|残差| > 2×1.244mm 视为标注噪声剔除）
  · MEK：log1p 变换 + 两阶段（P(MEK≥300) 边界概率作为回归特征）
  · XGB(0.85) + LGBM(0.15) 加权集成
  · 评估：5 折 CV + 体系留出（LOSO）双口径
"""
import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier, DMatrix
from lightgbm import LGBMRegressor, LGBMClassifier

from data import TARGETS, SYSTEMS, mat_lib, samples_df, labeled, present_codes
from features import build_matrix, build_row, feature_names

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SEED = 42
MEK_CAP = 300.0
T_NOISE_STD = 1.244

PARAMS_XGB_R = dict(n_estimators=600, learning_rate=0.05, max_depth=5,
                    subsample=0.9, colsample_bytree=0.8, min_child_weight=3,
                    reg_lambda=1.5, random_state=SEED, tree_method="hist")
PARAMS_LGB_R = dict(n_estimators=600, learning_rate=0.05, num_leaves=24,
                    subsample=0.9, colsample_bytree=0.8, min_child_samples=5,
                    reg_lambda=1.5, random_state=SEED, verbose=-1)
PARAMS_XGB_C = dict(n_estimators=500, learning_rate=0.05, max_depth=4,
                    subsample=0.9, colsample_bytree=0.8, min_child_weight=3,
                    reg_lambda=1.5, random_state=SEED, tree_method="hist",
                    eval_metric="auc")
PARAMS_LGB_C = dict(n_estimators=500, learning_rate=0.05, num_leaves=16,
                    subsample=0.9, colsample_bytree=0.8, min_child_samples=5,
                    reg_lambda=1.5, random_state=SEED, verbose=-1)


def transform(tgt, y):
    return np.sqrt(np.clip(y, 0, None)) if tgt == "T弯" else np.log1p(np.clip(y, 0, None))


def restore(tgt, v):
    return np.clip(v, 0, None) ** 2 if tgt == "T弯" else np.clip(np.expm1(v), 0, 550)


def series_enc(tr_s, y_tr, te_s, m=5.0):
    g = float(np.nanmean(y_tr))
    d = pd.DataFrame({"s": np.asarray(tr_s), "y": y_tr}).groupby("s")["y"].agg(["mean", "count"])
    enc = (d["mean"] * d["count"] + g * m) / (d["count"] + m)
    tr = pd.Series(np.asarray(tr_s)).map(enc).fillna(g).values
    te = pd.Series(np.asarray(te_s)).map(enc).fillna(g).values
    return tr, te, enc.to_dict(), g


def _fit_reg(X, y, kind):
    if kind == "lgb":
        return LGBMRegressor(**PARAMS_LGB_R).fit(X, y)
    return XGBRegressor(**PARAMS_XGB_R).fit(X, y)


def _fit_clf(X, y, kind):
    if kind == "lgb":
        return LGBMClassifier(**PARAMS_LGB_C).fit(X, y)
    return XGBClassifier(**PARAMS_XGB_C).fit(X, y)


def _blend(px, pl, w=0.85):
    return w * np.asarray(px) + (1 - w) * np.asarray(pl)


def _t弯_filter_inner(Xtr, ytr, str_):
    """T弯噪声过滤：训练集内 OOF 残差超阈值的样本剔除后返回保留索引。"""
    keep = np.ones(len(ytr), dtype=bool)
    for itr, ite in KFold(5, shuffle=True, random_state=SEED).split(Xtr):
        est = _fit_reg(Xtr[itr], ytr[itr], "xgb")
        r = ytr[ite] - est.predict(Xtr[ite])
        keep[ite] = np.abs(r) <= 2 * T_NOISE_STD
    return keep


def _p_hi_oof(Xtr, y_tr_raw, str_):
    """训练集内 OOF 边界概率（MEK 两阶段特征）。"""
    yb = (np.asarray(y_tr_raw) >= MEK_CAP).astype(int)
    if len(set(yb)) < 2:
        return np.full(len(yb), float(np.mean(yb)))
    p = np.zeros(len(yb))
    for itr, ite in KFold(5, shuffle=True, random_state=SEED).split(Xtr):
        c = _fit_clf(Xtr[itr], yb[itr], "xgb")
        p[ite] = c.predict_proba(Xtr[ite])[:, 1]
    return p


class CoatingPredictor:
    """三目标预测器；fit 在全量数据上产出最终模型，cv/loso 用于诚实评估。"""

    def __init__(self, codes=None, models=None, encoders=None, meta=None):
        self.codes = codes
        self.models = models or {}
        self.encoders = encoders or {}
        self.meta = meta or {}

    # ---------------- 内部：单目标管道 ----------------
    def _pipeline_oof(self, X0, series, y_raw, tgt, splits):
        y_t = transform(tgt, y_raw)
        oof = np.zeros(len(y_t))
        for tr, te in splits:
            se_tr, se_te, _, _ = series_enc(series[tr], y_t[tr], series[te])
            Xtr = np.column_stack([X0[tr], se_tr])
            Xte = np.column_stack([X0[te], se_te])
            if tgt == "T弯":
                keep = _t弯_filter_inner(Xtr, y_t[tr], series[tr])
                Xtr, ytr = Xtr[keep], y_t[tr][keep]
            elif tgt == "MEK":
                p_hi_tr = _p_hi_oof(Xtr, y_raw[tr], series[tr])
                yb_tr = (np.asarray(y_raw)[tr] >= MEK_CAP).astype(int)
                c = _fit_clf(Xtr, yb_tr, "xgb")
                Xte = np.column_stack([Xte, c.predict_proba(Xte)[:, 1]])
                Xtr = np.column_stack([Xtr, p_hi_tr])
                ytr = y_t[tr]
            else:
                ytr = y_t[tr]
            px = _fit_reg(Xtr, ytr, "xgb").predict(Xte)
            pl = _fit_reg(Xtr, ytr, "lgb").predict(Xte)
            oof[te] = _blend(px, pl)
        return oof

    def _pipeline_final(self, X0, series, y_raw, tgt):
        y_t = transform(tgt, y_raw)
        # 全量 OOF 系列编码（防泄漏）；未知系列的预测期中性值 = 训练目标全局均值
        se = np.zeros(len(y_t))
        g = float(np.mean(y_t))
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X0):
            _, se_te, _, g = series_enc(series[tr], y_t[tr], series[te])
            se[te] = se_te
        Xf = np.column_stack([X0, se])
        extra = {"series_global": g}
        if tgt == "T弯":
            keep = _t弯_filter_inner(Xf, y_t, series)
            Xf, yf = Xf[keep], y_t[keep]
        elif tgt == "MEK":
            yb_all = (np.asarray(y_raw) >= MEK_CAP).astype(int)
            extra["boundary"] = _fit_clf(Xf, yb_all, "xgb")
            p_hi = _p_hi_oof(Xf, y_raw, series)
            Xf = np.column_stack([Xf, p_hi])
            yf = y_t
        else:
            yf = y_t
        mx = _fit_reg(Xf, yf, "xgb")
        ml = _fit_reg(Xf, yf, "lgb")
        return dict(xgb=mx, lgb=ml, **extra)

    # ---------------- 训练 ----------------
    @classmethod
    def train(cls, verbose=True):
        mlib = mat_lib()
        df = labeled(samples_df())
        codes = present_codes(df)
        X = build_matrix(df, mlib, codes).values
        series = df["系列"].values
        p = cls(codes=codes)
        p.meta["cv"] = {}
        for tgt in ["T弯", "MEK"]:
            m = df[tgt].notna().values
            y_raw = df.loc[m, tgt].values.astype(float)
            p.models[tgt] = p._pipeline_final(X[m], series[m], y_raw, tgt)
            p.meta[f"{tgt}_series_global"] = p.models[tgt]["series_global"]
        m = df["水煮"].notna().values
        yb = (df.loc[m, "水煮"] >= 4).astype(int).values
        se = np.zeros(len(yb))
        y_all = df.loc[m, "水煮"].values.astype(float)
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X[m]):
            _, se_te, _, _ = series_enc(series[m][tr], y_all[tr], series[m][te])
            se[te] = se_te
        Xf = np.column_stack([X[m], se])
        p.models["水煮"] = dict(xgb=_fit_clf(Xf, yb, "xgb"), lgb=_fit_clf(Xf, yb, "lgb"))
        p.meta["水煮_series_global"] = float(np.mean(y_all))
        if verbose:
            n_t, n_m, n_w = int(df["T弯"].notna().sum()), int(df["MEK"].notna().sum()), int(m.sum())
            print(f"训练完成: T弯 {n_t}条 / MEK {n_m}条 / 水煮 {n_w}条(正例 {int(yb.sum())})")
        return p

    # ---------------- 预测 ----------------
    def _features(self, comp, system, bake_temp, bake_time, series=None):
        mlib = mat_lib()
        row = np.array([build_row(comp, system, bake_temp, bake_time, mlib, self.codes)], dtype=float)
        return row, series

    def predict(self, comp, system, bake_temp, bake_time, series=None, explain=False):
        row, _ = self._features(comp, system, bake_temp, bake_time, series)
        out = {}
        for tgt in ["T弯", "MEK"]:
            mdl = self.models[tgt]
            se_val = self.meta.get(f"{tgt}_series_global", 0.0)
            Xf = np.column_stack([row, [se_val]])
            if tgt == "MEK":
                pb = mdl["boundary"].predict_proba(Xf)[:, 1]
                Xf = np.column_stack([Xf, pb])
            px = mdl["xgb"].predict(Xf)[0]
            pl = mdl["lgb"].predict(Xf)[0]
            out[tgt] = float(restore(tgt, _blend([px], [pl])[0]))
        cx, cl = self.models["水煮"]["xgb"], self.models["水煮"]["lgb"]
        se_val = self.meta.get("水煮_series_global", 0.0)
        Xf = np.column_stack([row, [se_val]])
        out["水煮≥4概率"] = float(0.85 * cx.predict_proba(Xf)[0, 1] + 0.15 * cl.predict_proba(Xf)[0, 1])
        if explain:
            out["_drivers"] = self.drivers(row, comp)
        return out

    def drivers(self, row, comp, top=4):
        exp = {}
        for tgt in ["T弯", "MEK"]:
            mdl = self.models[tgt]
            se_val = self.meta.get(f"{tgt}_series_global", 0.0)
            Xf = np.column_stack([row, [se_val]])
            if tgt == "MEK":
                pb = mdl["boundary"].predict_proba(Xf)[:, 1]
                Xf = np.column_stack([Xf, pb])
            dm = DMatrix(Xf)
            contribs = mdl["xgb"].get_booster().predict(dm, pred_contribs=True)[0]
            names = feature_names(self.codes) + ["系列编码"] + (["P(MEK≥300)"] if tgt == "MEK" else [])
            order = np.argsort(-np.abs(contribs[:-1]))[:top]
            exp[tgt] = [(names[i], float(contribs[i])) for i in order]
        return exp

    # ---------------- 评估 ----------------
    def cv_evaluate(self, k=5, verbose=True):
        mlib = mat_lib()
        df = labeled(samples_df())
        X = build_matrix(df, mlib, self.codes).values
        series = df["系列"].values
        res = {}
        for tgt in ["T弯", "MEK"]:
            m = df[tgt].notna().values
            y = df.loc[m, tgt].values.astype(float)
            splits = list(KFold(k, shuffle=True, random_state=SEED).split(X[m]))
            oof = self._pipeline_oof(X[m], series[m], y, tgt, splits)
            pv = restore(tgt, oof)
            res[tgt] = dict(n=int(m.sum()), r2=float(r2_score(y, pv)),
                            mae=float(mean_absolute_error(y, pv)))
        m = df["水煮"].notna().values
        yb = (df.loc[m, "水煮"] >= 4).astype(int).values
        y_all = df.loc[m, "水煮"].values.astype(float)
        oofp = np.zeros(len(yb))
        for tr, te in KFold(k, shuffle=True, random_state=SEED).split(X[m]):
            se_tr, se_te, _, _ = series_enc(series[m][tr], y_all[tr], series[m][te])
            Xtr = np.column_stack([X[m][tr], se_tr])
            Xte = np.column_stack([X[m][te], se_te])
            px = _fit_clf(Xtr, yb[tr], "xgb").predict_proba(Xte)[:, 1]
            pl = _fit_clf(Xtr, yb[tr], "lgb").predict_proba(Xte)[:, 1]
            oofp[te] = 0.85 * px + 0.15 * pl
        res["水煮"] = dict(n=int(m.sum()), auc=float(roc_auc_score(yb, oofp)),
                           acc=float(np.mean((oofp >= 0.5).astype(int) == yb)))
        self.meta["cv"] = res
        if verbose:
            print("== 5折交叉验证（全体系，含系列OOF编码+集成）==")
            for tgt, v in res.items():
                if tgt == "水煮":
                    print(f"  水煮≥4: n={v['n']} AUC={v['auc']:.3f} Acc={v['acc']:.3f}")
                else:
                    print(f"  {tgt}: n={v['n']} R²={v['r2']:.3f} MAE={v['mae']:.2f}")
        return res

    def loso_evaluate(self, verbose=True):
        mlib = mat_lib()
        df = labeled(samples_df())
        X = build_matrix(df, mlib, self.codes).values
        series = df["系列"].values
        res = {}
        for held in SYSTEMS:
            te = (df["体系"] == held).values
            tr = ~te
            r = {"n_test": int(te.sum())}
            for tgt in ["T弯", "MEK"]:
                m_tr = tr & df[tgt].notna().values
                m_te = te & df[tgt].notna().values
                if m_te.sum() < 5 or m_tr.sum() < 20:
                    r[tgt] = None
                    continue
                y_raw = df.loc[m_tr, tgt].values.astype(float)
                y_t = transform(tgt, y_raw)
                se_tr, se_te, _, _ = series_enc(series[m_tr], y_t, series[m_te])
                Xtr = np.column_stack([X[m_tr], se_tr])
                Xte = np.column_stack([X[m_te], se_te])
                if tgt == "T弯":
                    keep = _t弯_filter_inner(Xtr, y_t, series[m_tr])
                    Xtr, ytr = Xtr[keep], y_t[keep]
                elif tgt == "MEK":
                    yb_all = (y_raw >= MEK_CAP).astype(int)
                    c = _fit_clf(Xtr, yb_all, "xgb")
                    Xte = np.column_stack([Xte, c.predict_proba(Xte)[:, 1]])
                    p_hi = _p_hi_oof(Xtr, y_raw, series[m_tr])
                    Xtr = np.column_stack([Xtr, p_hi])
                    ytr = y_t
                else:
                    ytr = y_t
                pv = restore(tgt, _blend(_fit_reg(Xtr, ytr, "xgb").predict(Xte),
                                         _fit_reg(Xtr, ytr, "lgb").predict(Xte)))
                r[tgt] = dict(mae=float(mean_absolute_error(df.loc[m_te, tgt].values, pv)),
                              r2=float(r2_score(df.loc[m_te, tgt].values, pv)))
            m_te = te & df["水煮"].notna().values
            m_trw = tr & df["水煮"].notna().values
            if m_te.sum() >= 5 and m_trw.sum() >= 20:
                y_all_tr = df.loc[m_trw, "水煮"].values.astype(float)
                yb_tr = (df.loc[m_trw, "水煮"] >= 4).astype(int).values
                se_tr, se_te, _, _ = series_enc(series[m_trw], y_all_tr, series[m_te])
                c = _fit_clf(np.column_stack([X[m_trw], se_tr]), yb_tr, "xgb")
                pp = c.predict_proba(np.column_stack([X[m_te], se_te]))[:, 1]
                yb_te = (df.loc[m_te, "水煮"] >= 4).astype(int).values
                r["水煮"] = dict(auc=float(roc_auc_score(yb_te, pp)) if len(set(yb_te)) > 1 else None,
                                 acc=float(np.mean((pp >= 0.5).astype(int) == yb_te)))
            res[held] = r
        if verbose:
            print("== 体系留出（训练不含该体系）==")
            for held, r in res.items():
                line = f"  {held}(n={r['n_test']}): "
                for tgt in ["T弯", "MEK"]:
                    if r.get(tgt):
                        line += f"{tgt} MAE={r[tgt]['mae']:.2f} "
                if r.get("水煮") and r["水煮"].get("acc") is not None:
                    line += f"水煮Acc={r['水煮']['acc']:.2f}"
                print(line)
        return res

    # ---------------- 持久化 ----------------
    def save(self):
        import joblib
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({"models": self.models, "codes": self.codes, "meta": self.meta},
                    os.path.join(MODEL_DIR, "predictor.joblib"))

    @classmethod
    def load(cls):
        import joblib
        d = joblib.load(os.path.join(MODEL_DIR, "predictor.joblib"))
        return cls(codes=d["codes"], models=d["models"], meta=d["meta"])


if __name__ == "__main__":
    p = CoatingPredictor.train()
    p.cv_evaluate()
    p.loso_evaluate()
    p.save()
