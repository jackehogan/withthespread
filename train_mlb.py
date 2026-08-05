"""
Train the MLB cover-probability model and save it to data/mlb_model.pkl.

Trains on seasons up to and including EVAL_SEASON - 1, tunes hyperparameters
on EVAL_SEASON, then saves the bundle.  The saved bundle is loaded by:
  - _kelly_analysis.py   (backtest / Kelly sizing report)
  - predict_mlb.py       (daily prediction pipeline)

Usage
-----
    python train_mlb.py              # default 50 hyperopt evals
    python train_mlb.py --evals 100  # more thorough search
"""
import argparse

import pandas as pd

import db
import data_pipeline as dp
import model as ml
from config import MLB

EVAL_SEASON  = 2025
NEXT_PERIOD  = 163
SPLIT_PERIOD = MLB.eval_split_period

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train the MLB cover model")
parser.add_argument(
    "--evals", type=int, default=50,
    help="Number of Hyperopt TPE evaluations (default 50)",
)
args = parser.parse_args()

# ── Load & filter data ───────────────────────────────────────────────────────
print("Loading data...")
client = db.connect()
try:
    all_games = db.fetch_games(client, "mlb")
finally:
    client.close()

seasons = sorted(all_games["season"].unique())
all_games = pd.concat(
    [dp.filter_regular_season(all_games[all_games["season"] == s], MLB, s)
     for s in seasons],
    ignore_index=True,
)

games_phase1 = all_games[all_games["season"] <= EVAL_SEASON].copy()
train_seasons = sorted(s for s in games_phase1["season"].unique() if s != EVAL_SEASON)

print(f"  Training seasons: {train_seasons}  |  Tune season: {EVAL_SEASON}")
print(f"  Rows: {len(games_phase1)}")

# ── Train ─────────────────────────────────────────────────────────────────────
clf, scores, best_lookback, best_k, style_model, win_clf = ml.train_models(
    games_phase1, NEXT_PERIOD, EVAL_SEASON, SPLIT_PERIOD, max_evals=args.evals,
)

# ── Save ──────────────────────────────────────────────────────────────────────
ml.save_model(
    clf, scores, best_lookback, best_k, style_model,
    next_period=NEXT_PERIOD, train_seasons=train_seasons, win_clf=win_clf,
)

print(f"\nDone.")
print(f"  lookback={best_lookback}, Elo K={best_k}, "
      f"bp_ip_14d={ml._BP_FATIGUE_DAYS}d (fixed)")
print(f"  Scores: {scores}")
