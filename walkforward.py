"""
Rolling-origin (walk-forward) evaluation for the MLB win model.

train_mlb.py uses one fixed split: fit 2022-24, tune on 2025, and 2025 is also
the only number reported. That makes `win_val_roc` a selection score, not a
holdout score -- it reads 0.632 while the honest 2026 figure is 0.616 -- and it
burns a whole season that could be training data.

Walk-forward fixes both. Each fold:

    fit seasons        -> fit the model
    tune season        -> choose hyperparameters (never fitted on)
    score season       -> touched once, after everything is decided

  fold   fit        tune    score
    A    2022       2023    2024
    B    2022-23    2024    2025
    C    2022-24    2025    2026

No season is ever both tuner and scorer, so the pooled score column is honest
by construction, and you get several estimates instead of one -- which turns
"is 0.616 real?" into a question with error bars.

Predictions are symmetrised per game exactly as production does, so the numbers
are comparable to what the pipeline actually emits.

    python walkforward.py                 # all folds
    python walkforward.py --folds 2026    # just the last one
"""
import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

import db
import model as ml

NEXT_PERIOD = 163
SEARCH = 25          # random hyperparameter draws per fold


def _sample(rng):
    return {
        "n_estimators":     int(rng.choice([100, 150, 200, 300])),
        "max_depth":        int(rng.choice([2, 3, 4])),
        "learning_rate":    float(rng.choice([0.05, 0.1, 0.15, 0.22])),
        "min_child_weight": float(rng.uniform(1.0, 8.0)),
        "reg_alpha":        float(rng.uniform(0.0, 6.0)),
        "reg_lambda":       float(rng.uniform(0.0, 6.0)),
        "subsample":        float(rng.choice([0.7, 0.85, 1.0])),
        "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
    }


def _fit(params, X, y):
    m = XGBClassifier(**params, eval_metric="logloss", verbosity=0,
                      enable_categorical=True)
    m.fit(X, y)
    return m


def _symmetrise(p, pk):
    p = pd.Series(p, index=pk.index)
    s = p.groupby(pk).transform("sum")
    return (p + (1 - s) / 2).clip(1e-6, 1 - 1e-6)


def build(games, score_season, k):
    sub = games[games["season"] <= score_season]
    cache = ml._precompute(sub, next_period=NEXT_PERIOD,
                           eval_season=score_season, k_values=[k])
    out = ml.build_features(sub, next_period=NEXT_PERIOD, lookback=7,
                            eval_season=score_season, eval_split_period=1,
                            best_k=k, _cache=cache)
    X_tr, _, _, _, X_val, _ = out[:6]
    wl = out[7]
    feats = [c for c in X_tr.columns if c != "team"]

    tr = X_tr.reset_index(drop=True).copy()
    tr["_season"] = [key[1] for key in wl["keys_train"]]
    tr["_y"] = pd.Series(wl["train"]).reset_index(drop=True).to_numpy()

    va = X_val.reset_index(drop=True).copy()
    va["_y"] = pd.Series(wl["val"]).reset_index(drop=True).to_numpy()
    gg = games.dropna(subset=["opponent"]).copy()
    gg["season"] = pd.to_numeric(gg["season"], errors="coerce")
    gg["period"] = pd.to_numeric(gg["period"], errors="coerce")
    k2 = gg.set_index(["team", "season", "period"])["game_pk"].to_dict()
    va["_pk"] = [k2.get(key, np.nan) for key in wl["keys_val"]]
    # The 5am moneyline the pipeline actually bets into. odds_source is
    # odds-api-historical at 08:55Z on every row, five minutes before the
    # nightly runs, so this is a price that was really on the board.
    k2o = gg.set_index(["team", "season", "period"])["ml_odds"].to_dict()
    va["_ml"] = pd.to_numeric(
        pd.Series([k2o.get(key, np.nan) for key in wl["keys_val"]]), errors="coerce")
    return tr.dropna(subset=["_y"]), va.dropna(subset=["_y", "_pk"]), feats


def roi_table(y, p, ml, thresholds=(0.0, 0.05, 0.10, 0.20)):
    """Flat-stake ROI at each EV threshold, one unit risked per bet."""
    ok = np.isfinite(ml)
    y, p, ml = y[ok], p[ok], ml[ok]
    pay = np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))
    ev = p * pay - (1 - p)
    out = []
    for t in thresholds:
        sel = ev > t
        if sel.sum() < 25:
            continue
        pnl = np.where(y[sel] == 1, pay[sel], -1.0)
        out.append({"thr": t, "n": int(sel.sum()), "win": float(y[sel].mean()),
                    "roi": float(pnl.sum() / sel.sum()), "pnl": pnl})
    return out


def fold(games, score_season, k, rng):
    tr, va, feats = build(games, score_season, k)
    seasons = sorted(tr["_season"].unique())
    tune = seasons[-1]
    fit_s = seasons[:-1]
    if not fit_s:
        print(f"  fold {score_season}: no fit seasons before the tuner — skipped")
        return None
    A = tr[tr["_season"].isin(fit_s)]
    B = tr[tr["_season"] == tune]
    print(f"  fit {fit_s} ({len(A)} rows) | tune {tune} ({len(B)} rows) "
          f"| score {score_season} ({len(va)} rows)")

    best, best_auc = None, -1
    for _ in range(SEARCH):
        p = _sample(rng)
        m = _fit(p, A[feats], A["_y"])
        a = roc_auc_score(B["_y"], m.predict_proba(B[feats])[:, 1])
        if a > best_auc:
            best, best_auc = p, a
    print(f"    tuned on {tune}: AUC {best_auc:.4f}  (depth {best['max_depth']}, "
          f"{best['n_estimators']} trees)")

    # refit on fit + tune, then touch the score season once
    m = _fit(best, tr[feats], tr["_y"])
    p = _symmetrise(m.predict_proba(va[feats])[:, 1], va["_pk"])
    y = va["_y"].to_numpy()
    roi = roi_table(y, p.to_numpy(), va["_ml"].to_numpy())
    for r in roi:
        print(f"      EV>{r['thr']:.2f}: {r['n']:>5} bets  win {r['win']*100:5.1f}%"
              f"  ROI {r['roi']*100:+6.1f}%")
    return {"score_season": score_season, "n": len(va),
            "tune_auc": best_auc,
            "auc": roc_auc_score(y, p), "logloss": log_loss(y, p),
            "brier": brier_score_loss(y, p), "params": best, "roi": roi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=[2024, 2025, 2026])
    ap.add_argument("--k", type=float, default=ml._K_CANDIDATES[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-params", default=None,
                    help="write the chosen hyperparameters per fold to this JSON")
    args = ap.parse_args()

    client = db.connect()
    try:
        games = db.fetch_games(client, "mlb")
    finally:
        client.close()
    rng = np.random.default_rng(args.seed)

    rows = []
    for s in args.folds:
        print(f"\n=== fold: score {s} ===")
        r = fold(games, s, args.k, rng)
        if r:
            rows.append(r)
            print(f"    HOLDOUT {s}: AUC {r['auc']:.4f}  logloss {r['logloss']:.4f}"
                  f"  brier {r['brier']:.4f}")

    if not rows:
        return
    print("\n" + "=" * 66)
    print(f"{'score season':<14}{'n':>7}{'tuned AUC':>12}{'HOLDOUT AUC':>14}"
          f"{'logloss':>10}{'brier':>9}")
    for r in rows:
        print(f"{r['score_season']:<14}{r['n']:>7}{r['tune_auc']:>12.4f}"
              f"{r['auc']:>14.4f}{r['logloss']:>10.4f}{r['brier']:>9.4f}")
    a = np.array([r["auc"] for r in rows])
    w = np.array([r["n"] for r in rows], dtype=float)
    print(f"\npooled holdout AUC (n-weighted): {np.average(a, weights=w):.4f}")
    print(f"across folds: mean {a.mean():.4f}  sd {a.std(ddof=1) if len(a)>1 else float('nan'):.4f}")
    if args.dump_params:
        # The production fit takes its hyperparameters from here rather than
        # from scoring EVAL_SEASON, so the season being predicted never
        # influences the choice.
        with open(args.dump_params, "w", encoding="utf-8") as f:
            json.dump({str(r["score_season"]): {"params": r["params"],
                                                "tune_auc": r["tune_auc"],
                                                "holdout_auc": r["auc"]}
                       for r in rows}, f, indent=2)
        print("")
        print(f"wrote {args.dump_params}")

    print("")
    print("FLAT-STAKE ROI on untouched seasons (5am moneyline, 1u per bet)")
    print(f"{'EV threshold':<14}" + "".join(f"{r['score_season']:>12}" for r in rows)
          + f"{'pooled':>12}{'95% CI':>20}")
    boot_rng = np.random.default_rng(1)
    for t in (0.0, 0.05, 0.10, 0.20):
        cells, pnls = [], []
        for r in rows:
            hit = next((x for x in r["roi"] if x["thr"] == t), None)
            cells.append(f"{hit['roi']*100:+11.1f}%" if hit else f"{'--':>12}")
            if hit is not None:
                pnls.append(hit["pnl"])
        if not pnls:
            continue
        allp = np.concatenate(pnls)
        boot = [boot_rng.choice(allp, len(allp), replace=True).mean() * 100
                for _ in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{'EV>' + format(t, '.2f'):<14}" + "".join(cells)
              + f"{allp.mean()*100:+11.1f}%"
              + f"   [{lo:+.1f}%, {hi:+.1f}%]".rjust(20))
    print("  n per fold: " + ", ".join(
        f"{r['score_season']}={next((x['n'] for x in r['roi'] if x['thr']==0.0), 0)}"
        for r in rows))

    gap = np.mean([r["tune_auc"] - r["auc"] for r in rows])
    print(f"mean tune-minus-holdout gap: {gap:+.4f}  "
          "(how optimistic a single-split number is)")


if __name__ == "__main__":
    main()
