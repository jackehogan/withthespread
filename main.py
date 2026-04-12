"""
Entry point for the sport-agnostic betting prediction pipeline.

All logic lives in the submodules:
  config.py        — SportConfig dataclass + NFL/NBA instances
  db.py            — MongoDB connection and CRUD helpers
  data_pipeline.py — API calls for game results and spreads
  model.py         — Feature engineering, XGBoost training, prediction
  pipeline.py      — Weekly run() and historical seed_season()

Usage
-----
    # One-time: create MongoDB indexes
    python main.py setup

    # One-time: backfill historical data (paid Odds API key required for spreads)
    python main.py seed --sport nba --seasons 2022 2023 2024
    python main.py seed --sport nfl --seasons 2020 2021 2022 2023

    # Weekly: run the prediction pipeline
    python main.py run --sport nba --season 2025 --period 55 --lookback 10
    python main.py run --sport nfl --season 2024 --period 12 --lookback 5
"""

import argparse
import datetime

import db
from config import SPORTS
from pipeline import run, seed_season


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_setup(args: argparse.Namespace) -> None:
    """Create MongoDB indexes (safe to run multiple times)."""
    client = db.connect()
    try:
        db.create_indexes(client)
    finally:
        client.close()


def _cmd_seed(args: argparse.Namespace) -> None:
    """Backfill historical game data for one or more seasons."""
    sport = SPORTS[args.sport]
    client = db.connect()
    try:
        for season in sorted(args.seasons):
            seed_season(client, sport, season, request_delay=args.delay)
    finally:
        client.close()
    print("\nSeeding complete.")


def _cmd_run(args: argparse.Namespace) -> None:
    """Run the weekly prediction pipeline and print results."""
    sport = SPORTS[args.sport]
    client = db.connect()
    try:
        result = run(
            sport=sport,
            season=args.season,
            next_period=args.period,
            client=client,
            key_type=args.key_type,
            max_evals=args.max_evals,
        )
        if not result.empty:
            print("\nPredictions:")
            print(result[["opponent", "spread", "predspread", "coverprob"]].to_string())
    finally:
        client.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sports betting spread prediction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    sub.add_parser("setup", help="Create MongoDB indexes (run once).")

    # seed
    p_seed = sub.add_parser("seed", help="Backfill historical game + spread data.")
    p_seed.add_argument("--sport", choices=list(SPORTS), required=True,
                        help="Sport to seed.")
    p_seed.add_argument("--seasons", type=int, nargs="+", required=True,
                        help="Season start years to backfill (e.g. 2022 2023 2024).")
    p_seed.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between paid API calls (default: 1.0).")

    # run
    p_run = sub.add_parser("run", help="Run the weekly prediction pipeline.")
    p_run.add_argument("--sport", choices=list(SPORTS), required=True,
                       help="Sport to run.")
    p_run.add_argument("--season", type=int, default=datetime.datetime.now().year,
                       help="Season start year (default: current year).")
    p_run.add_argument("--period", type=int, required=True,
                       help="Next period to predict (NFL week or NBA game number).")
    p_run.add_argument("--key-type", choices=["free", "paid"], default="free",
                       dest="key_type", help="Odds API key type (default: free).")
    p_run.add_argument("--max-evals", type=int, default=10, dest="max_evals",
                       help="Hyperopt budget per model (default: 10).")

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    {"setup": _cmd_setup, "seed": _cmd_seed, "run": _cmd_run}[args.command](args)
