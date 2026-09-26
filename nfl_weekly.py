"""
Weekly NFL job: refresh results, pull the Tuesday odds, score the coming week, email the bet slip.

Steps
  1. Download the nflverse schedule/results -> data/nflverse_games.csv
  2. Find the coming week (earliest regular-season week with a game today or later)
  3. Pull any missing Odds API snapshots (backfill_nfl_odds.py; cached, ~40 credits a week)
  4. Score every week of the season so far with predict_nfl_lookup.py (main + shadow model)
  5. Email: this week's bets, last week's results, season-to-date record for both models

Email settings come from the "email" block in data/config.txt (same as the MLB job).

    python nfl_weekly.py                     # full run
    python nfl_weekly.py --dry-run           # no odds pull, no email; writes logs/nfl_weekly_email.html
    python nfl_weekly.py --season 2026 --week 3 --dry-run
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import argparse
import json
import smtplib
import subprocess
import tempfile
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
GAMES = Path("data/nflverse_games.csv")
PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, help="override the season (default: from the schedule)")
ap.add_argument("--week", type=int, help="override the week to score (default: the coming week)")
ap.add_argument("--dry-run", action="store_true", help="skip the odds pull and the email; save the email HTML")
ap.add_argument("--skip-refresh", action="store_true", help="use the cached nflverse file")
ap.add_argument("--skip-odds", action="store_true", help="don't pull odds (use what is cached)")
ap.add_argument("--test", action="store_true", help="prefix the subject with [TEST]")
a = ap.parse_args()


def log(msg):
    print(f"[{datetime.now(ET):%H:%M:%S}] {msg}", flush=True)


# ---- 1. results --------------------------------------------------------------------------------
if not a.skip_refresh:
    r = requests.get(GAMES_URL, timeout=60)
    r.raise_for_status()
    GAMES.write_bytes(r.content)
    log(f"nflverse games refreshed ({len(r.content) / 1e6:.1f} MB)")

g = pd.read_csv(GAMES)
g = g[g.game_type == "REG"]

# ---- 2. which week -------------------------------------------------------------------------------
today = datetime.now(ET).date().isoformat()
if a.season and a.week:
    season, week = a.season, a.week
else:
    upcoming = g[g.gameday >= today].sort_values(["season", "week"])
    if upcoming.empty:
        log("no upcoming regular-season games - offseason, nothing to do")
        sys.exit(0)
    season, week = int(upcoming.season.iloc[0]), int(upcoming.week.iloc[0])
    season = a.season or season
log(f"coming week: {season} week {week}")

# ---- 3. odds -------------------------------------------------------------------------------------
odds_note = ""
if not (a.dry_run or a.skip_odds):
    cmd = [PY, "backfill_nfl_odds.py", "--seasons", *map(str, range(2020, season + 1)), "--apply"]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(out.stdout[-2000:], out.stderr[-2000:])
    if out.returncode != 0:
        odds_note = "Odds pull failed - lines below may fall back to the closing line at -110."
o = pd.read_parquet("data/nfl_odds_snapshots.parquet")
wk_games = set(g[(g.season == season) & (g.week == week)].game_id)
have_tue = set(o[(o.role == "tuesday") & (o.market == "spreads") & (o.book == "fanduel")].game_id) & wk_games
if wk_games and len(have_tue) < len(wk_games):
    odds_note += f" FanDuel Tuesday line missing for {len(wk_games) - len(have_tue)} of {len(wk_games)} games (consensus/close used)."
log(f"FanDuel Tuesday lines: {len(have_tue)}/{len(wk_games)} games")


# ---- 4. score the season so far -----------------------------------------------------------------------
def score(w):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "out.json"
        out = subprocess.run([PY, "predict_nfl_lookup.py", "--season", str(season), "--week", str(w), "--json", str(path)],
                             capture_output=True, text=True, encoding="utf-8")
        if out.returncode != 0 or not path.exists():
            log(f"week {w}: not scored ({(out.stdout + out.stderr).strip().splitlines()[-1:]})")
            return None
        return json.loads(path.read_text(encoding="utf-8"))


weeks = {w: score(w) for w in range(2, min(week, 17) + 1)}
weeks = {w: v for w, v in weeks.items() if v}
this = weeks.get(week)
last = weeks.get(week - 1)
log(f"scored weeks: {sorted(weeks)}")


# ---- 5. email -------------------------------------------------------------------------------------
def record(bets):
    bets = [b for b in bets if b["result"]]
    w_ = sum(b["result"] == "WIN" for b in bets)
    l_ = sum(b["result"] == "LOSS" for b in bets)
    p_ = sum(b["result"] == "PUSH" for b in bets)
    staked = sum(b["units"] for b in bets)
    pnl = sum(b["pnl"] for b in bets)
    return {"n": len(bets), "rec": f"{w_}-{l_}" + (f"-{p_}" if p_ else ""), "staked": staked, "pnl": pnl,
            "roi": 100 * pnl / staked if staked else 0.0}


TD = 'style="padding:4px 8px;border-bottom:1px solid #ddd;text-align:{al};white-space:nowrap"'
TH = 'style="padding:4px 8px;border-bottom:2px solid #333;text-align:{al};background:#f3f3f3"'


def table(rows, cols):
    """cols: list of (header, key or callable, align)"""
    if not rows:
        return "<p><i>none</i></p>"
    h = "".join(f"<th {TH.format(al=al)}>{escape(hd)}</th>" for hd, _, al in cols)
    body = ""
    for r in rows:
        cells = ""
        for _, k, al in cols:
            v = k(r) if callable(k) else r.get(k, "")
            cells += f"<td {TD.format(al=al)}>{v}</td>"
        body += f"<tr>{cells}</tr>"
    return f'<table style="border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px">{h and "<tr>" + h + "</tr>"}{body}</table>'


def res_cell(b):
    if not b["result"]:
        return "pending"
    col = {"WIN": "#1a7f37", "LOSS": "#c62828", "PUSH": "#666"}[b["result"]]
    return f'<b style="color:{col}">{b["result"]}</b> ({b["pnl"]:+.1f})'


pct = lambda x: f"{100 * x:.1f}%"
MAIN_COLS = [("Bet on", lambda b: f"<b>{escape(b['bet'])}</b>", "left"), ("Game", "matchup", "left"),
             ("Kickoff", "kickoff", "left"), ("Price", lambda b: f"{b['price']:+d}", "right"),
             ("Units", lambda b: f"<b>{b['units']:.1f}</b>", "right"), ("P(cover)", lambda b: pct(b["p_cover"]), "right"),
             ("Break-even", lambda b: pct(b["break_even"]), "right"), ("Similar games", lambda b: f"{b['similar']:.0f}", "right"),
             ("Situation", "situation", "left"), ("Bet team last game", "reason_team", "left"),
             ("Opponent last game", "reason_opp", "left")]
SHADOW_COLS = [("Bet on", lambda b: f"<b>{escape(b['bet'])}</b>", "left"), ("Game", "matchup", "left"),
               ("Price", lambda b: f"{b['price']:+d}", "right"), ("Units", lambda b: f"{b['units']:.1f}", "right"),
               ("P(cover)", lambda b: pct(b["p_cover"]), "right"), ("Similar games", lambda b: f"{b['similar']:.0f}", "right"),
               ("Corrections (us / opp)", "corrections", "left"), ("Also main bet?", lambda b: "yes" if b["also_main"] else "NO", "left")]
RESULT_COLS = [("Bet on", "bet", "left"), ("Game", "matchup", "left"), ("Price", lambda b: f"{b['price']:+d}", "right"),
               ("Units", lambda b: f"{b['units']:.1f}", "right"), ("Result", res_cell, "left")]

parts = [f'<div style="font-family:Arial,sans-serif;font-size:14px;color:#222">',
         f"<h2 style='margin-bottom:4px'>NFL {season} &mdash; week {week}</h2>"]
if odds_note:
    parts.append(f"<p style='color:#c62828'>{escape(odds_note.strip())}</p>")

if week in (1, 18) or this is None:
    why = {1: "Week 1 has no previous game this season, so the model doesn't bet it.",
           18: "Week 18 is not bet (teams rest starters)."}.get(week, "This week could not be scored - see the job log.")
    parts.append(f"<p><b>No bets this week.</b> {why}</p>")
    main_now, shadow_now = [], []
else:
    main_now, shadow_now = this["main"], this["shadow"] or []
    units = sum(b["units"] for b in main_now)
    parts.append(f"<h3 style='margin-bottom:4px'>This week's bets: {len(main_now)} bet{'s' * (len(main_now) != 1)}, "
                 f"{units:.1f} units</h3>")
    parts.append("<p style='margin-top:0;color:#555'>Lines and prices are FanDuel's Tuesday 10:00 ET numbers. "
                 "1 unit per point of cover probability above break-even, cap 8, bets of 0.6+ units.</p>")
    parts.append(table(main_now, MAIN_COLS))
    parts.append("<p style='color:#555;font-size:12px'>Similar games = past team-games that looked like this one. "
                 "2020-25: bets with under 25 similar games (RARE) went 64-26 for +161 units; the rest were about break-even.</p>")
    parts.append("<h4 style='margin-bottom:4px'>Shadow model (not bet): n-2 correction x 0.5</h4>")
    parts.append(table(shadow_now, SHADOW_COLS))

if last:
    rm, rs = record(last["main"]), record(last["shadow"] or [])
    parts.append(f"<h3 style='margin-bottom:4px'>Last week (week {week - 1}): {rm['rec']}, {rm['pnl']:+.1f} units</h3>")
    parts.append(table(last["main"], RESULT_COLS))
    parts.append(f"<p style='color:#555'>Shadow model last week: {rs['rec']}, {rs['pnl']:+.1f} units</p>")

rows = []
for w, v in sorted(weeks.items()):
    rm, rs = record(v["main"]), record(v["shadow"] or [])
    if rm["n"] or rs["n"]:
        rows.append({"week": w, "m": rm, "s": rs})
if rows:
    tm = record([b for v in weeks.values() for b in v["main"]])
    ts = record([b for v in weeks.values() for b in (v["shadow"] or [])])
    cum = 0.0
    for r in rows:
        cum += r["m"]["pnl"]
        r["cum"] = cum
    parts.append(f"<h3 style='margin-bottom:4px'>Season to date: {tm['rec']}, {tm['pnl']:+.1f} units, ROI {tm['roi']:.1f}%</h3>")
    parts.append(table(rows, [("Week", "week", "right"), ("Record", lambda r: r["m"]["rec"], "right"),
                              ("Staked", lambda r: f"{r['m']['staked']:.1f}", "right"),
                              ("Profit", lambda r: f"{r['m']['pnl']:+.1f}", "right"),
                              ("Running total", lambda r: f"{r['cum']:+.1f}", "right"),
                              ("Shadow record", lambda r: r["s"]["rec"], "right"),
                              ("Shadow profit", lambda r: f"{r['s']['pnl']:+.1f}", "right")]))
    parts.append(f"<p style='color:#555'>Shadow model season to date: {ts['rec']}, {ts['pnl']:+.1f} units, "
                 f"ROI {ts['roi']:.1f}%</p>")

cfg_model = (this or last or {})
parts.append(f"<p style='color:#888;font-size:11px'>Model: 4-input lookup (both teams' last-game spread and spreadscore), "
             f"trained {cfg_model.get('train', '')}, widths {cfg_model.get('widths', '')}. "
             f"Walk-forward 2020-25: about 53 bets a season, 20% ROI. Generated {datetime.now(ET):%a %b %d %H:%M} ET.</p></div>")
html = "\n".join(parts)

n_bets = len(main_now)
subject = (f"NFL week {week}: {n_bets} bet{'s' * (n_bets != 1)}" + (f", {sum(b['units'] for b in main_now):.1f} units" if n_bets else "")
           + (f" | last week {record(last['main'])['pnl']:+.1f}u" if last else ""))

subject = ("[TEST] " if a.test else "") + subject
Path("logs").mkdir(exist_ok=True)
Path("logs/nfl_weekly_email.html").write_text(html, encoding="utf-8")
log(f"subject: {subject}")
if a.dry_run:
    log("dry run - email saved to logs/nfl_weekly_email.html, not sent")
    sys.exit(0)

cfg = json.load(open("data/config.txt")).get("email")
if not cfg:
    sys.exit("no email block in data/config.txt")
msg = MIMEMultipart("alternative")
msg["Subject"], msg["From"], msg["To"] = subject, cfg["username"], cfg["to"]
msg.attach(MIMEText(html, "html", "utf-8"))
with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"])) as server:
    server.starttls()
    server.login(cfg["username"], cfg["app_password"])
    server.sendmail(cfg["username"], [x.strip() for x in cfg["to"].split(",")], msg.as_string())
log(f"email sent to {cfg['to']}")
