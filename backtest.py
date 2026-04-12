"""
Backtest the NBA spread-cover prediction model.

For each period in a given season, trains on all prior data,
generates predictions, then compares against actual outcomes.
Reports accuracy bucketed by confidence level.

Usage
-----
    python backtest.py --sport nba --season 2024 --lookback 10 --start-period 20
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
    lookback: int,
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
    records = []

    for period in range(start_period, max_period + 1):
        # Ground truth: did the team cover in this period?
        truth = season_games[season_games["period"] == period][
            ["team", "spreadscore"]
        ].dropna()
        if truth.empty:
            continue
        truth["covered"] = (truth["spreadscore"] > 0).astype(int)

        # Build features using all data available before this period
        try:
            X_train, X_test, y_train, y_test, X_val, y_val = ml.build_features(
                all_games, period, lookback, sport.eval_season, sport.eval_split_period
            )
            reg, clas, _ = ml.train_models(
                X_train, X_test, y_train, y_test, X_val, y_val, max_evals
            )
            X_pred = ml.build_prediction_features(
                season_games, period, lookback, eval_season
            )
        except Exception as exc:
            print(f"  Period {period}: skipped ({exc})")
            continue

        if X_pred.empty:
            continue

        preds = ml.predict(reg, clas, X_pred)

        merged = preds.join(truth.set_index("team")[["covered"]], how="inner")
        merged["period"] = period
        records.append(merged)

        n = len(merged)
        acc = (merged["coverprob"].round() == merged["covered"]).mean()
        print(f"  Period {period:3d}: {n:2d} teams, acc={acc:.3f}, "
              f"coverprob range [{merged['coverprob'].min():.3f}, {merged['coverprob'].max():.3f}]")

    if not records:
        print("No predictions generated.")
        return pd.DataFrame()

    results = pd.concat(records)
    _print_summary(results)
    return results


def _print_summary(results: pd.DataFrame) -> None:
    results = results.copy()
    results["predicted_cover"] = (results["coverprob"] >= 0.5).astype(int)
    results["correct"] = (results["predicted_cover"] == results["covered"]).astype(int)

    overall_acc = results["correct"].mean()
    n = len(results)
    print(f"\n{'='*50}")
    print(f"Overall accuracy: {overall_acc:.3f}  ({n} predictions)")
    print(f"{'='*50}")

    # Bucket by confidence
    bins = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 1.01]
    labels = ["<0.45 (high fade)", "0.45-0.50", "0.50-0.55", "0.55-0.60",
              "0.60-0.65", ">0.65 (high cover)"]
    results["bucket"] = pd.cut(results["coverprob"], bins=bins, labels=labels, right=False)

    print("\nAccuracy by confidence bucket:")
    print(f"{'Bucket':<22} {'N':>5}  {'Acc':>6}  {'Cover%':>7}")
    print("-" * 44)
    for label in labels:
        g = results[results["bucket"] == label]
        if g.empty:
            continue
        acc = g["correct"].mean()
        cover_rate = g["covered"].mean()
        print(f"{label:<22} {len(g):>5}  {acc:>6.3f}  {cover_rate:>7.3f}")

    print(f"\nBase cover rate: {results['covered'].mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest spread cover predictions.")
    parser.add_argument("--sport", choices=list(SPORTS), default="nba")
    parser.add_argument("--season", type=int, required=True,
                        help="Season to evaluate (e.g. 2024)")
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--start-period", type=int, default=20, dest="start_period",
                        help="First period to predict (need lookback prior periods)")
    parser.add_argument("--max-evals", type=int, default=10, dest="max_evals",
                        help="Hyperopt budget per model per period (keep low for speed)")
    args = parser.parse_args()

    run_backtest(args.sport, args.season, args.lookback, args.start_period, args.max_evals)
