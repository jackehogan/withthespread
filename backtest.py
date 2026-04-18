"""
Backtest the NBA spread-cover prediction model.

For each period in a given season, trains on all prior data,
generates predictions, then compares against actual outcomes.
Reports accuracy bucketed by confidence level.

Usage
-----
    python backtest.py --sport nba --season 2024 --start-period 20
"""

import argparse

import numpy as np
import pandas as pd

import db
import model as ml
from config import SPORTS
from pipeline import _fetch_games_filtered


def run_backtest(
    sport_name: str,
    eval_season: int,
    start_period: int,
    max_evals: int,
) -> pd.DataFrame:
    sport = SPORTS[sport_name]
    client = db.connect()

    try:
        all_games = _fetch_games_filtered(client, sport)
        season_games = _fetch_games_filtered(client, sport, eval_season)
    finally:
        client.close()

    if all_games.empty or season_games.empty:
        print("No data found.")
        return pd.DataFrame()

    max_period = int(season_games["period"].max())
    records      = []   # per-prediction rows
    diagnostics  = []   # per-period model stats
    feat_records = []   # feature matrices

    for period in range(start_period, max_period + 1):
        # Ground truth
        truth = season_games[season_games["period"] == period][
            ["team", "spreadscore", "opponent"]
        ].dropna(subset=["spreadscore"])
        truth = truth[truth["spreadscore"] != 0]
        if truth.empty:
            continue
        truth["covered"] = (truth["spreadscore"] > 0).astype(int)

        try:
            reg, sigma_diff, scores, best_lookback, best_k, style_model = ml.train_models(
                all_games, period, eval_season, sport.eval_split_period, max_evals
            )

            # Build upcoming_context from actual game data for this period.
            # Populates home, is_b2b, spread, opponent — same as live prediction time.
            period_rows = season_games[season_games["period"] == period]
            if not period_rows.empty:
                uc = period_rows.set_index("team")[
                    [c for c in ["home", "spread", "opponent", "date"] if c in period_rows.columns]
                ].copy()
                if "date" in uc.columns:
                    prev_rows = season_games[season_games["period"] < period]
                    last_date = prev_rows.groupby("team")["date"].max()
                    rest = (
                        pd.to_datetime(uc["date"])
                        - pd.to_datetime(uc.index.map(last_date))
                    ).dt.days
                    uc["is_b2b"] = (rest == 1).astype(float)
                    uc = uc.drop(columns=["date"])
                else:
                    uc["is_b2b"] = np.nan
                upcoming_context = uc
            else:
                upcoming_context = None

            X_pred = ml.build_prediction_features(
                season_games, period, best_lookback, eval_season,
                upcoming_context, style_model, best_k
            )
        except Exception as exc:
            print(f"  Period {period}: skipped ({exc})")
            continue

        if X_pred.empty:
            continue

        # Build opponent map from upcoming_context (or fall back to ground truth)
        if upcoming_context is not None and "opponent" in upcoming_context.columns:
            opponent_map = upcoming_context["opponent"].dropna().to_dict()
        else:
            opponent_map = truth.set_index("team")["opponent"].dropna().to_dict()

        preds = ml.predict(reg, sigma_diff, X_pred, opponent_map)

        truth_indexed = truth.set_index("team")[["covered", "spreadscore"]]
        merged = preds.join(truth_indexed, how="inner")

        # Attach model params to every row for downstream slicing
        merged["period"]     = period
        merged["lb"]         = best_lookback
        merged["elo_k"]      = best_k
        merged["sigma_diff"] = round(sigma_diff, 3)
        records.append(merged)

        # Per-period diagnostics (includes full scores dict from train_models)
        n    = len(merged)
        acc  = (merged["coverprob"].round() == merged["covered"]).mean()
        mae  = (merged["predspread"] - merged["spreadscore"]).abs().mean()
        corr = merged["predspread"].corr(merged["spreadscore"])
        cp_min, cp_max = merged["coverprob"].min(), merged["coverprob"].max()
        diagnostics.append({
            "period":        period,
            "n_teams":       n,
            "lb":            best_lookback,
            "elo_k":         best_k,
            "sigma_diff":    round(sigma_diff, 3),
            "acc":           round(float(acc),  4),
            "mae":           round(float(mae),  4),
            "corr":          round(float(corr), 4),
            "cp_min":        round(float(cp_min), 4),
            "cp_max":        round(float(cp_max), 4),
            "train_r2":      scores.get("reg_train_r2"),
            "test_r2":       scores.get("reg_test_r2"),
            "val_r2":        scores.get("reg_val_r2"),
            "train_sign":    scores.get("reg_train_sign"),
            "test_sign":     scores.get("reg_test_sign"),
            "sigma_source":  scores.get("sigma_source"),
        })

        # Feature matrix — indexed by team, tagged with period for joining later
        feat_frame = X_pred.copy()
        feat_frame["period"] = period
        feat_frame = feat_frame.drop(columns=["team"], errors="ignore")
        feat_frame.index.name = "team"
        feat_records.append(feat_frame.reset_index())

        print(f"  Period {period:3d}: {n:2d} teams, lb={best_lookback}, "
              f"acc={acc:.3f}, coverprob [{cp_min:.3f}, {cp_max:.3f}]  "
              f"MAE={mae:.2f} corr={corr:.3f}  sigma={sigma_diff:.2f}")

    if not records:
        print("No predictions generated.")
        return pd.DataFrame()

    results = pd.concat(records)
    results.index.name = "team"
    results = results.reset_index()

    prefix = f"backtest_{sport_name}_{eval_season}"

    pred_path = f"{prefix}_predictions.csv"
    results.to_csv(pred_path, index=False)
    print(f"\nPredictions  -> {pred_path}")

    diag_path = f"{prefix}_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False)
    print(f"Diagnostics  -> {diag_path}")

    feat_path = f"{prefix}_features.csv"
    pd.concat(feat_records, ignore_index=True).to_csv(feat_path, index=False)
    print(f"Features     -> {feat_path}")

    _print_summary(results)

    return results


def _print_summary(results: pd.DataFrame) -> None:
    results = results.copy().reset_index(drop=True)
    results["predicted_cover"] = (results["coverprob"] >= 0.5).astype(int)
    results["correct"]   = (results["predicted_cover"] == results["covered"]).astype(int)
    results["reg_error"] = results["predspread"] - results["spreadscore"]

    n           = len(results)
    overall_acc = results["correct"].mean()
    overall_mae = results["reg_error"].abs().mean()
    overall_corr= results["predspread"].corr(results["spreadscore"])
    sign_acc    = (np.sign(results["predspread"]) == np.sign(results["spreadscore"])).mean()
    pct_above   = (results["coverprob"] > 0.5).mean()

    print(f"\n{'='*62}")
    print(f"Overall accuracy:      {overall_acc:.3f}  ({n} predictions)")
    print(f"Sign accuracy:         {sign_acc:.3f}  (predspread direction correct)")
    print(f"Regression MAE:        {overall_mae:.3f}  SpreadScore units")
    print(f"Regression corr:       {overall_corr:.3f}  (pred vs actual)")
    print(f"Predictions > 0.5:     {pct_above:.1%}  (should be ~50%)")
    print(f"{'='*62}")

    # Bucket by confidence
    bins   = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.01]
    labels = ["<0.40 (strong fade)", "0.40-0.45", "0.45-0.50",
              "0.50-0.55", "0.55-0.60", ">0.60 (strong cover)"]
    results["bucket"] = pd.cut(results["coverprob"], bins=bins, labels=labels, right=False)

    print("\nAccuracy by confidence bucket:")
    print(f"  {'Bucket':<22} {'N':>5}  {'Acc':>6}  {'Cover%':>7}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*6}  {'-'*7}")
    for label in labels:
        g = results[results["bucket"] == label]
        if g.empty:
            continue
        acc       = g["correct"].mean()
        cover_rate= g["covered"].mean()
        print(f"  {label:<22} {len(g):>5}  {acc:>6.3f}  {cover_rate:>7.3f}")

    print(f"\n  Base cover rate: {results['covered'].mean():.3f}")

    # Regression error distribution
    print(f"\nRegression error (predspread - actual SpreadScore):")
    print(f"  Bias (mean):  {results['reg_error'].mean():+.3f}")
    print(f"  Std:          {results['reg_error'].std():.3f}")
    print(f"  |err| <= 3:   {(results['reg_error'].abs() <= 3).mean():.1%}")
    print(f"  |err| <= 7:   {(results['reg_error'].abs() <= 7).mean():.1%}")
    print(f"  |err| > 15:   {(results['reg_error'].abs() > 15).mean():.1%}")

    # spread_diff distribution
    if "spread_diff" in results.columns:
        print(f"\nspread_diff (predspread_A - predspread_B):")
        print(f"  Mean:  {results['spread_diff'].mean():+.3f}  (should be ~0)")
        print(f"  Std:   {results['spread_diff'].std():.3f}")
        print(f"  Range: [{results['spread_diff'].min():.2f}, {results['spread_diff'].max():.2f}]")

    # Decile analysis: equal-count bins by coverprob
    print(f"\nAccuracy by coverprob decile (equal-count bins, ~{n//10} predictions each):")
    print(f"  {'Decile (coverprob range)':<28} {'N':>5}  {'Acc':>6}  {'Cover%':>7}  {'Edge':>6}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}")
    results["decile"] = pd.qcut(results["coverprob"], q=10, duplicates="drop")
    base = results["covered"].mean()
    for interval, g in results.groupby("decile", observed=True):
        acc  = g["correct"].mean()
        covr = g["covered"].mean()
        edge = acc - base
        lo, hi = interval.left, interval.right
        print(f"  {lo:.3f} – {hi:.3f}               {len(g):>5}  {acc:>6.3f}  {covr:>7.3f}  {edge:>+6.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest spread cover predictions.")
    parser.add_argument("--sport", choices=list(SPORTS), default="nba")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start-period", type=int, default=20, dest="start_period")
    parser.add_argument("--max-evals", type=int, default=10, dest="max_evals")
    args = parser.parse_args()

    run_backtest(args.sport, args.season, args.start_period, args.max_evals)
