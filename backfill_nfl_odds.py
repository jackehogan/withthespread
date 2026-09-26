"""
Backfill NFL spreads + totals from The Odds API historical endpoint.

Two snapshots per NFL week, chosen so each game gets three lines:

    sun : Sunday 12:30 ET, before the 1pm kickoffs
          -> LOOKAHEAD line for next week's games (set before this week is played)
          -> NEAR-CLOSE line for this week's not-yet-started games
    tue : Tuesday 10:00 ET
          -> TUESDAY line for the coming week (after the market reacted)

Snapshot times come from the nflverse schedule: for week w, the Tuesday is the
one on/before the week's first game and the Sunday is two days before that.
One extra Sunday after the last Tuesday gives the final week's near-close.
Future timestamps are skipped, so re-running mid-season picks up new weeks.

Every raw response is cached under data/odds_api_nfl/<season>/, so the run is
resumable and nothing is paid for twice. Cost: 10 credits per market per
region per call -> 20 per call here.

Output: data/nfl_odds_snapshots.parquet, one row per (snapshot, game, book,
market), matched to nflverse game_id, with provenance columns:
    odds_source, pull, pull_week, requested_ts, snapshot_ts, book, market,
    role (lookahead | near_close | tuesday | other),
    home_point / home_price / away_price   (spreads; home_point in nflverse
                                            convention: + = home favored)
    total_point / over_price / under_price (totals)

    python backfill_nfl_odds.py                 # dry run: list pulls + cost
    python backfill_nfl_odds.py --apply         # pull missing snapshots
    python backfill_nfl_odds.py --flatten-only  # rebuild parquet from cache
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
BASE = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds"
MARKETS = "spreads,totals"
CREDITS_PER_CALL = 20
CACHE_DIR = Path("data/odds_api_nfl")
OUT = Path("data/nfl_odds_snapshots.parquet")

ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Football Team": "WAS",
    "Washington Commanders": "WAS", "Washington Redskins": "WAS",
}

parser = argparse.ArgumentParser()
parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2020, 2027)))
parser.add_argument("--apply", action="store_true")
parser.add_argument("--flatten-only", action="store_true")
parser.add_argument("--max-credits", type=int, default=4800)
parser.add_argument("--reserve", type=int, default=5000,
                    help="stop if the account balance would drop below this")
parser.add_argument("--delay", type=float, default=0.3)
args = parser.parse_args()

nv = pd.read_csv("data/nflverse_games.csv")
nv = nv[(nv.game_type == "REG") & nv.season.isin(args.seasons)].copy()
nv["gameday"] = pd.to_datetime(nv.gameday).dt.date


def pulls_for(season: int) -> list[dict]:
    at = lambda d, hh, mm: datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)
    out = []
    weeks = nv[nv.season == season].groupby("week").gameday.min().sort_index()
    for w, first in weeks.items():
        tue = first - timedelta(days=(first.weekday() - 1) % 7)
        out.append({"season": season, "pull": "sun", "pull_week": int(w), "ts": at(tue - timedelta(days=2), 12, 30)})
        out.append({"season": season, "pull": "tue", "pull_week": int(w), "ts": at(tue, 10, 0)})
    last_tue = out[-1]["ts"].date()
    out.append({"season": season, "pull": "sun", "pull_week": int(weeks.index.max()) + 1,
                "ts": at(last_tue + timedelta(days=5), 12, 30)})
    now = datetime.now(UTC)
    return [p for p in out if p["ts"] < now]


def cache_path(p: dict) -> Path:
    return CACHE_DIR / str(p["season"]) / f"{p['pull']}_w{p['pull_week']:02d}.json"


pulls = [p for s in args.seasons for p in pulls_for(s)]
todo = [p for p in pulls if not cache_path(p).exists()]
print(f"seasons {args.seasons}: {len(pulls)} snapshots, {len(pulls) - len(todo)} cached, "
      f"{len(todo)} to pull -> {len(todo) * CREDITS_PER_CALL:,} credits "
      f"(cap {args.max_credits:,}, reserve {args.reserve:,})")

if args.apply and not args.flatten_only:
    key = json.load(open("data/config.txt"))["spreads"]["key_paid"]
    spent, remaining = 0, None
    for i, p in enumerate(todo, 1):
        if spent + CREDITS_PER_CALL > args.max_credits:
            print(f"credit cap reached after {i - 1} pulls")
            break
        if remaining is not None and remaining - CREDITS_PER_CALL < args.reserve:
            print(f"reserve reached (balance {remaining}) after {i - 1} pulls")
            break
        iso = p["ts"].astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = requests.get(BASE, params={"apiKey": key, "regions": "us", "markets": MARKETS,
                                       "oddsFormat": "american", "date": iso}, timeout=30)
        if r.status_code != 200:
            print(f"  {iso} HTTP {r.status_code}: {r.text[:200]}")
            break
        spent += int(r.headers.get("x-requests-last", CREDITS_PER_CALL))
        remaining = int(r.headers.get("x-requests-remaining", 0))
        path = cache_path(p)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"requested_ts": iso, **{k: v for k, v in p.items() if k != "ts"},
                                    "resp": r.json()}))
        if i % 20 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {p['season']} {p['pull']} w{p['pull_week']}  "
                  f"spent {spent:,}  balance {remaining:,}")
        time.sleep(args.delay)

# ---- flatten -------------------------------------------------------------------
games = nv[["game_id", "season", "week", "gameday", "home_team", "away_team"]]
gkey = {(r.home_team, r.away_team, r.gameday): (r.game_id, r.season, r.week) for r in games.itertuples()}
rows, unmapped, unmatched = [], set(), 0
for path in sorted(CACHE_DIR.glob("*/*.json")):
    c = json.loads(path.read_text())
    if c["season"] not in args.seasons:
        continue
    snap_ts = c["resp"].get("timestamp")
    for ev in c["resp"].get("data", []):
        h, a = ABBR.get(ev["home_team"]), ABBR.get(ev["away_team"])
        if not h or not a:
            unmapped |= {ev["home_team"], ev["away_team"]} - set(ABBR)
            continue
        kick = pd.Timestamp(ev["commence_time"]).tz_convert(ET)
        g = gkey.get((h, a, kick.date()))
        if g is None:
            unmatched += 1
            continue
        game_id, season, week = g
        if c["pull"] == "tue" and week == c["pull_week"]:
            role = "tuesday"
        elif c["pull"] == "sun" and week == c["pull_week"]:
            role = "lookahead"
        elif c["pull"] == "sun" and week == c["pull_week"] - 1:
            role = "near_close"
        else:
            role = "other"
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                rec = {"game_id": game_id, "season": season, "week": week,
                       "home_team": h, "away_team": a, "commence_time": ev["commence_time"],
                       "odds_source": "odds_api_historical", "pull": c["pull"],
                       "pull_week": c["pull_week"], "role": role,
                       "requested_ts": c["requested_ts"], "snapshot_ts": snap_ts,
                       "book": bk["key"], "book_last_update": m.get("last_update"),
                       "market": m["key"]}
                o = {x["name"]: x for x in m["outcomes"]}
                if m["key"] == "spreads":
                    hh, aa = o.get(ev["home_team"]), o.get(ev["away_team"])
                    if not hh or hh.get("point") is None:
                        continue
                    rec.update(home_point=-float(hh["point"]), home_price=hh.get("price"),
                               away_price=aa.get("price") if aa else None)
                elif m["key"] == "totals":
                    ov, un = o.get("Over"), o.get("Under")
                    if not ov or ov.get("point") is None:
                        continue
                    rec.update(total_point=float(ov["point"]), over_price=ov.get("price"),
                               under_price=un.get("price") if un else None)
                rows.append(rec)

df = pd.DataFrame(rows)
if df.empty:
    print("no cached snapshots to flatten")
    sys.exit()
df.to_parquet(OUT, index=False)
print(f"\nwrote {OUT}: {len(df):,} rows; events not matched to a REG game: {unmatched}"
      + (f"; unmapped names: {sorted(unmapped)}" if unmapped else ""))

# ---- coverage ------------------------------------------------------------------
n_games = nv.groupby("season").size().rename("games")
cov = {}
for role in ("lookahead", "tuesday", "near_close"):
    for mkt in ("spreads", "totals"):
        d = df[(df.role == role) & (df.market == mkt)]
        cov[f"{role}_{mkt}_any"] = d.groupby("season").game_id.nunique()
        cov[f"{role}_{mkt}_fd"] = d[d.book == "fanduel"].groupby("season").game_id.nunique()
cov = pd.DataFrame(cov).reindex(n_games.index).fillna(0)
pct = cov.div(n_games, axis=0).mul(100).round(0).astype(int)
print("\n% of regular-season games with a line (any book / FanDuel):")
print(pd.concat([n_games, pct], axis=1).to_string())
