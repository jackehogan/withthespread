"""
Test A: truncation test for within-season lookahead.

A causal feature for (team, season, period P) must not change when the season
is cut off before P. Rebuild features with 2026 truncated at date D -- prior
seasons stay whole, since when predicting a 2026 game you legitimately know all
of 2022-2025 -- then compare every 2026 row before D against the same rows from
the full-data build.

Any feature whose value moves saw the future. This catches the current game's
own outcome bleeding into a rolling window, opponent lookups resolving forward
(which happened here once already), post-game Elo, and centred windows.

    python _audit_lookahead.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib

import db
import model as ml

CUTOFFS = ["2026-06-01", "2026-07-15", "2026-08-15"]
NEXT_PERIOD = 163


def build(games, k, lb):
    cache = ml._precompute(games, next_period=NEXT_PERIOD,
                           eval_season=2026, k_values=[k])
    out = ml.build_features(games, next_period=NEXT_PERIOD, lookback=lb,
                            eval_season=2026, eval_split_period=1,
                            best_k=k, _cache=cache)
    X_val, wl = out[4], out[7]
    d = X_val.reset_index(drop=True).copy()
    d["_key"] = list(wl["keys_val"])
    return d.set_index("_key")


def main():
    b = joblib.load(r"data\mlb_model.pkl")
    # Use the LIVE feature set and the live K, not the bundle's -- the bundle
    # predates the Elo restore and still lists fast_elo_diff.
    feats = list(ml._KEEP_FEATURES)
    k, lb = ml._K_CANDIDATES[0], b["best_lookback"]

    client = db.connect()
    try:
        g = db.fetch_games(client, "mlb")
    finally:
        client.close()
    g["date"] = pd.to_datetime(g["date"])

    print("building full-data reference...")
    full = build(g, k, lb)
    print(f"  reference rows: {len(full)}")

    print(f"\n{'cutoff':<14}{'rows cmp':>10}  per-feature mismatches")
    any_bad = False
    for cut in CUTOFFS:
        cd = pd.Timestamp(cut)
        gt = g[(g["season"] < 2026) | ((g["season"] == 2026) & (g["date"] < cd))]
        tr = build(gt, k, lb)
        common = full.index.intersection(tr.index)
        # only rows genuinely before the cutoff
        keep = [key for key in common
                if g[(g["team"] == key[0]) & (g["season"] == key[1])
                     & (g["period"] == key[2])]["date"].min() < cd]
        if not keep:
            print(f"{cut:<14}{0:>10}  (no comparable rows)")
            continue
        bad = {}
        for f in feats:
            if f not in full.columns or f not in tr.columns:
                bad[f] = "ABSENT"
                continue
            a = pd.to_numeric(full.loc[keep, f], errors="coerce")
            c = pd.to_numeric(tr.loc[keep, f], errors="coerce")
            both = a.notna() & c.notna()
            diff = (a[both] - c[both]).abs()
            tol = np.maximum(1e-6, 1e-4 * a[both].abs())
            n_bad = int((diff > tol).sum())
            # a feature going from present to absent is also a change
            n_bad += int((a.notna() & c.isna()).sum())
            if n_bad:
                bad[f] = f"{n_bad}/{len(keep)} (max delta {diff.max():.4g})"
        if bad:
            any_bad = True
            print(f"{cut:<14}{len(keep):>10}  LOOKAHEAD SUSPECTED")
            for f, why in bad.items():
                print(f"{'':26}{f:<26}{why}")
        else:
            print(f"{cut:<14}{len(keep):>10}  all {len(feats)} features identical")

    print("\n" + ("FAIL - at least one feature changed under truncation"
                  if any_bad else
                  "PASS - every feature is invariant to truncating the future"))


if __name__ == "__main__":
    main()
