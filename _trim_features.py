"""
Backward feature elimination for the MLB win model.

Selection is scored by leave-one-season-out CV **inside the training seasons
only**. 2026 is touched exactly once, at the end, as a clean read --
selecting on it would burn the only holdout that has never been used, which is
precisely what made 2025 unusable as an honest number.

Features are eliminated in mirrored UNITS: dropping sp_era without
opp_sp_era would leave the model a half-matchup, which is not a set anyone
would ship.

    python _trim_features.py
"""
import pickle
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

D = (r"C:\Users\Jack\AppData\Local\Temp\claude"
     r"\C--Users-Jack-OneDrive-Documents-GitHub-WithTheSpread"
     r"\6366ec4c-69fb-46b7-9d5c-78324ebcad57\scratchpad")
KEEP_PARAMS = ("n_estimators", "max_depth", "learning_rate", "min_child_weight",
               "reg_alpha", "reg_lambda", "subsample", "colsample_bytree", "gamma")


def units(feats):
    """Group each feature with its opp_ mirror so they are dropped together."""
    seen, out = set(), []
    for f in feats:
        if f in seen:
            continue
        m = "opp_" + f
        if m in feats:
            out.append((f, [f, m])); seen |= {f, m}
        elif f.startswith("opp_") and f[4:] in feats:
            continue
        else:
            out.append((f, [f])); seen.add(f)
    return out


def cv_auc(tr, cols, params):
    """Leave-one-season-out AUC across the training seasons."""
    scores = []
    for s in sorted(tr["_season"].unique()):
        a, b = tr[tr["_season"] != s], tr[tr["_season"] == s]
        m = XGBClassifier(**params, eval_metric="logloss", verbosity=0, enable_categorical=True)
        m.fit(a[cols], a["_y"])
        scores.append(roc_auc_score(b["_y"], m.predict_proba(b[cols])[:, 1]))
    return float(np.mean(scores))


def holdout_auc(tr, ho, cols, params):
    m = XGBClassifier(**params, eval_metric="logloss", verbosity=0, enable_categorical=True)
    m.fit(tr[cols], tr["_y"])
    p = pd.Series(m.predict_proba(ho[cols])[:, 1], index=ho.index)
    s = p.groupby(ho["_pk"]).transform("sum")          # symmetrise, as production does
    p = (p + (1 - s) / 2).clip(1e-6, 1 - 1e-6)
    return roc_auc_score(ho["_y"], p)


def main():
    with open(D + r"\matrices2.pkl", "rb") as f:
        M = pickle.load(f)
    tr, ho, feats = M["train"], M["holdout"], M["features"]
    params = {k: v for k, v in M["params"].items() if k in KEEP_PARAMS and v is not None}
    ho = ho.dropna(subset=["_pk"])
    print(f"train {len(tr)} rows (seasons {sorted(tr['_season'].unique())})")
    print(f"holdout {len(ho)} rows (2026)   features {len(feats)}")
    print(f"params: {params}\n")

    cols = list(feats)
    base = cv_auc(tr, cols, params)
    print(f"full set: CV {base:.4f}")
    history = [(len(cols), base, None)]

    while True:
        us = units(cols)
        if len(us) <= 6:
            break
        best = None
        for name, group in us:
            trial = [c for c in cols if c not in group]
            sc = cv_auc(tr, trial, params)
            if best is None or sc > best[1]:
                best = (name, sc, group)
        name, sc, group = best
        if sc < history[-1][1] - 0.0015:      # stop once removal really costs
            print(f"\nstop: dropping {name} would cost {history[-1][1]-sc:.4f}")
            break
        cols = [c for c in cols if c not in group]
        history.append((len(cols), sc, name))
        print(f"  drop {name:<22} -> {len(cols):>2} feats, CV {sc:.4f} "
              f"({sc-base:+.4f} vs full)")

    print("\n" + "=" * 62)
    print(f"{'set':<28}{'n':>4}{'CV(22-24)':>12}{'2026 holdout':>14}")
    full_h = holdout_auc(tr, ho, feats, params)
    trim_h = holdout_auc(tr, ho, cols, params)
    print(f"{'full':<28}{len(feats):>4}{base:>12.4f}{full_h:>14.4f}")
    print(f"{'trimmed':<28}{len(cols):>4}{history[-1][1]:>12.4f}{trim_h:>14.4f}")
    print("\nkept:")
    for u, _ in units(cols):
        print("   ", u)
    print("\ndropped:")
    for _, _, n in history[1:]:
        print("   ", n)
    with open(D + r"\trim_result.pkl", "wb") as f:
        pickle.dump({"kept": cols, "history": history}, f)


if __name__ == "__main__":
    main()
