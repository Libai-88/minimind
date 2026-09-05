# -*- coding: utf-8 -*-
"""涂料配方智能系统命令行入口。

示例：
  python3 cli.py sample
  python3 cli.py evaluate
  python3 cli.py predict --system 环氧酚醛 --components "IR190=66,RF516=2.6,RF956=1.5,正丁醇=3,补加混合液=1.3" --bake 200x10
  python3 cli.py recommend --system 环氧酚醛 --target "T弯<=12" --target "MEK>=100" --target "水煮>=4"
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data import mat_lib, samples_df, labeled, SYSTEMS  # noqa: E402
from features import mech_readout  # noqa: E402
from knowledge import check_formulation, rule_summary, MATERIAL_POOL_NOTES  # noqa: E402
from predictor import CoatingPredictor  # noqa: E402

MODEL_PATH = os.path.join(_HERE, "models", "predictor.joblib")


def get_predictor(retrain=False):
    if not retrain and os.path.exists(MODEL_PATH):
        p = CoatingPredictor.load()
        p.meta.setdefault("cv", {})
        return p
    p = CoatingPredictor.train()
    p.cv_evaluate()
    p.loso_evaluate()
    p.save()
    return p


def parse_components(s):
    comp = {}
    for part in s.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k:
            comp[k] = float(v)
    total = sum(comp.values())
    if total > 0 and abs(total - 100) > 20:
        comp = {k: v / total * 100 for k, v in comp.items()}
        print(f"[提示] 组分总和 {total:g}%，已归一化为 100%")
    return comp


def parse_bake(s):
    if not s:
        return None
    t, _, m = s.partition("x")
    return float(t), float(m)


def cmd_sample(_):
    df = samples_df()
    lab = labeled(df)
    print(f"样本总数 {len(df)}（实测 {len(lab)}）")
    print(lab.groupby("体系").agg(n=("样本ID", "size"), T弯=("T弯", "mean"),
                                  MEK=("MEK", "mean"), 水煮=("水煮", "mean")).round(2))
    mlib = mat_lib()
    print(f"材料库 {len(mlib)} 种：", {r: sum(1 for v in mlib.values() if v['role'] == r)
                                     for r in ['树脂', '固化剂', '溶剂', '助剂', '颜料']})


def cmd_predict(args):
    mlib = mat_lib()
    comp = parse_components(args.components)
    bad = [k for k in comp if k not in mlib]
    if bad:
        print(f"[错误] 未知原料: {bad}。请使用项目材料库中的原料名（python3 cli.py sample 查看）")
        return
    system = args.system
    bake = parse_bake(args.bake)
    if bake:
        bt, btm = bake
    else:
        bt = btm = None
    mech = mech_readout(comp, mlib, bt, btm)
    results = check_formulation(comp, mlib, system, bt, btm, mech)
    errs, warns, oks = rule_summary(results)

    p = get_predictor(args.retrain)
    out = p.predict(comp, system, bt, btm, explain=True)

    print(f"\n== 性能预测 | {system} 体系 | 烘烤 {bt or '—'}°C×{btm or '—'}min ==")
    if "T弯" in out:
        cv = p.meta.get("cv", {}).get("T弯", {})
        print(f"  T弯   预测 {out['T弯']:6.1f} mm（越小柔韧性越好）"
              + (f"  | 历史CV误差 ±{cv.get('mae', 0):.1f} mm" if cv else ""))
    if "MEK" in out:
        cv = p.meta.get("cv", {}).get("MEK", {})
        print(f"  MEK   预测 {out['MEK']:6.0f} 次（越大耐溶剂性越好）"
              + (f"  | 历史CV误差 ±{cv.get('mae', 0):.0f} 次" if cv else ""))
    if "水煮≥4概率" in out:
        cv = p.meta.get("cv", {}).get("水煮", {})
        print(f"  水煮   预计 ≥4级概率 {out['水煮≥4概率'] * 100:5.1f}%（越大概率耐水煮越好）")
    print("\n[专家规则检查]")
    for lv, msg in results:
        mark = {"ok": "✓", "warn": "！", "error": "✗"}[lv]
        print(f"  {mark} {msg}")
    print("\n[机理读数]")
    for k in ["r_phenol_epoxy", "r_nco_oh", "ne_effective", "tg_fox_solids",
              "cure_margin", "solids_frac", "pvc"]:
        if k in mech and mech[k] is not None and not (isinstance(mech[k], float) and np.isnan(mech[k])):
            print(f"  {k} = {mech[k]:.3f}")
    if out.get("_drivers"):
        print("\n[关键驱动因素（SHAP 贡献）]")
        for tgt, drv in out["_drivers"].items():
            tops = "、".join(f"{n}({c:+.2f})" for n, c in drv)
            print(f"  {tgt}: {tops}")


def cmd_recommend(args):
    p = get_predictor(args.retrain)
    from recommender import Recommender
    rec = Recommender(p)
    bake = parse_bake(args.bake)
    rec.recommend(args.system, args.target, top_n=args.top, bake=bake,
                  n_iter=args.iter, seed=args.seed)


def cmd_evaluate(args):
    p = get_predictor(args.retrain)
    p.cv_evaluate()
    p.loso_evaluate()


def main():
    ap = argparse.ArgumentParser(description="涂料配方性能预测与推荐系统")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sample", help="数据概览")

    pe = sub.add_parser("evaluate", help="交叉验证与体系留出评估")
    pe.add_argument("--retrain", action="store_true")

    pp = sub.add_parser("predict", help="配方性能预测")
    pp.add_argument("--system", required=True, choices=SYSTEMS)
    pp.add_argument("--components", required=True, help='示例: "IR190=66,RF516=2.6"')
    pp.add_argument("--bake", default=None, help="如 200x10")
    pp.add_argument("--retrain", action="store_true")

    pr = sub.add_parser("recommend", help="按目标性能推荐配方")
    pr.add_argument("--system", required=True, choices=SYSTEMS)
    pr.add_argument("--target", action="append", required=True,
                    help='可多次，如 --target "T弯<=12" --target "MEK>=100"')
    pr.add_argument("--bake", default=None)
    pr.add_argument("--top", type=int, default=3)
    pr.add_argument("--iter", type=int, default=3000)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--retrain", action="store_true")

    args = ap.parse_args()
    {"sample": cmd_sample, "evaluate": cmd_evaluate,
     "predict": cmd_predict, "recommend": cmd_recommend}[args.cmd](args)


if __name__ == "__main__":
    main()
