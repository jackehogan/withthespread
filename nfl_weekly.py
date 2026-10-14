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
from scipy.stats import norm

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
    cmd = [PY, "backfill_nfl_odds.py", "--seasons", *map(str, range(2020, season + 1)), "--apply", "--reserve", "1000"]
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
# every table lists the biggest bets first
for v in weeks.values():
    for k in ("main", "shadow"):
        if v.get(k):
            v[k] = sorted(v[k], key=lambda b: -b["units"])
this = weeks.get(week)
last = weeks.get(week - 1)
log(f"scored weeks: {sorted(weeks)}")


# ---- 5. email -------------------------------------------------------------------------------------
# Expected vs skill. One curve links cover probability and margin past the spread:
#   margin = 13.2 x inverse-normal(P)   <->   P = normal-cdf(margin / 13.2)
# Expected result: stake x (P x pay - (1 - P)) with the model's P at bet time.
# Skill result: the same formula with the P implied by the ACTUAL margin, so a game that plays out
# exactly as projected scores exactly its expected result. Walk-forward 2020-25 benchmark:
# actual ROI 20.0%, expected ROI 11.7%, skill ROI 8.5%.
SIG_ATS = 13.2


def _pay(b):
    return 100 / -b["price"] if b["price"] < 0 else b["price"] / 100


def exp_pnl(b):
    p_ = b["p_cover"]
    return b["units"] * (p_ * _pay(b) - (1 - p_))


def skill_pnl(b):
    if b.get("ats_margin") is None:
        return 0.0
    p_ = float(norm.cdf(b["ats_margin"] / SIG_ATS))
    return b["units"] * (p_ * _pay(b) - (1 - p_))


def wavg(bets, key):
    w = [(b["units"], b[key]) for b in bets if b.get(key) is not None]
    tot = sum(u for u, _ in w)
    return sum(u * v for u, v in w) / tot if tot else float("nan")


def record(bets):
    bets = [b for b in bets if b["result"]]
    w_ = sum(b["result"] == "WIN" for b in bets)
    l_ = sum(b["result"] == "LOSS" for b in bets)
    p_ = sum(b["result"] == "PUSH" for b in bets)
    staked = sum(b["units"] for b in bets)
    pnl = sum(b["pnl"] for b in bets)
    return {"n": len(bets), "rec": f"{w_}-{l_}" + (f"-{p_}" if p_ else ""), "staked": staked, "pnl": pnl,
            "roi": 100 * pnl / staked if staked else 0.0,
            "win": 100 * w_ / (w_ + l_) if w_ + l_ else float("nan"),  # pushes excluded
            "exp": sum(exp_pnl(b) for b in bets),
            "exp_roi": 100 * sum(exp_pnl(b) for b in bets) / staked if staked else float("nan"),
            "skill": sum(skill_pnl(b) for b in bets),
            "skill_roi": 100 * sum(skill_pnl(b) for b in bets) / staked if staked else float("nan"),
            "proj": wavg(bets, "proj_margin"), "actual": wavg(bets, "ats_margin")}


fmt_pct = lambda v: "&ndash;" if v != v else f"{v:.1f}%"


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


def margin_txt(m):
    if m is None:
        return "&ndash;"
    return "push" if m == 0 else (f"beat by {m:g}" if m > 0 else f"missed by {-m:g}")


pct = lambda x: f"{100 * x:.1f}%"
MAIN_COLS = [("Bet on", lambda b: f"<b>{escape(b['bet'])}</b>", "left"), ("Game", "matchup", "left"),
             ("Kickoff", "kickoff", "left"), ("Price", lambda b: f"{b['price']:+d}", "right"),
             ("Units", lambda b: f"<b>{b['units']:.1f}</b>", "right"), ("P(cover)", lambda b: pct(b["p_cover"]), "right"),
             ("Break-even", lambda b: pct(b["break_even"]), "right"),
             ("Projected vs spread", lambda b: f"{b['proj_margin']:+.1f}", "right"),
             ("Similar games", lambda b: f"{b['similar']:.0f}", "right"),
             ("Situation", "situation", "left"), ("Bet team last game", "reason_team", "left"),
             ("Opponent last game", "reason_opp", "left")]
SHADOW_COLS = [("Bet on", lambda b: f"<b>{escape(b['bet'])}</b>", "left"), ("Game", "matchup", "left"),
               ("Price", lambda b: f"{b['price']:+d}", "right"), ("Units", lambda b: f"{b['units']:.1f}", "right"),
               ("P(cover)", lambda b: pct(b["p_cover"]), "right"), ("Similar games", lambda b: f"{b['similar']:.0f}", "right"),
               ("Corrections (us / opp)", "corrections", "left"), ("Also main bet?", lambda b: "yes" if b["also_main"] else "NO", "left")]
RESULT_COLS = [("Bet on", "bet", "left"), ("Game", "matchup", "left"), ("Price", lambda b: f"{b['price']:+d}", "right"),
               ("Units", lambda b: f"{b['units']:.1f}", "right"),
               ("Projected", lambda b: f"{b['proj_margin']:+.1f}", "right"),
               ("Actual vs spread", lambda b: margin_txt(b.get("ats_margin")), "right"),
               ("Result", res_cell, "left"),
               ("Expected", lambda b: f"{exp_pnl(b):+.2f}", "right"),
               ("Skill", lambda b: f"{skill_pnl(b):+.2f}" if b["result"] else "", "right")]

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
    parts.append(f"<h3 style='margin-bottom:4px'>Last week (week {week - 1}): {rm['rec']}, {rm['pnl']:+.1f} units "
                 f"&nbsp;|&nbsp; expected {rm['exp']:+.2f}, skill {rm['skill']:+.2f}</h3>")
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
    parts.append(f"<h3 style='margin-bottom:4px'>Season to date: {tm['rec']}, {tm['pnl']:+.1f} units, "
                 f"win {fmt_pct(tm['win'])}, ROI {fmt_pct(tm['roi'])} &nbsp;|&nbsp; expected ROI {fmt_pct(tm['exp_roi'])}, "
                 f"skill ROI {fmt_pct(tm['skill_roi'])}</h3>")
    parts.append(f"<p style='margin-top:0;color:#555'>Bets were projected to beat the spread by {tm['proj']:+.1f} pts on average; "
                 f"they actually came in at {tm['actual']:+.1f} pts (stake-weighted).<br>"
                 "All season numbers are pooled (total / total staked). <b>Expected</b> = what the model claimed at bet time. "
                 "<b>Skill</b> = the same claim re-scored with the cover probability implied by the actual margin past the spread "
                 "(a bet that plays out exactly as projected scores its expected value). Compare skill with expected; "
                 "actual ROI is the money.<br>"
                 "Walk-forward 2020-25 benchmark: about 60% wins, ROI 20%, expected ROI 11.7%, skill ROI 8.5%, "
                 "projected +2.8 / actual +2.8 pts.</p>")
    rows.append({"week": "<b>Season</b>", "m": tm, "s": ts, "cum": tm["pnl"], "total": True})
    B = lambda v, r: f"<b>{v}</b>" if r.get("total") else v
    parts.append(table(rows, [("Week", "week", "right"), ("Record", lambda r: B(r["m"]["rec"], r), "right"),
                              ("Win %", lambda r: B(fmt_pct(r["m"]["win"]), r), "right"),
                              ("Staked", lambda r: B(f"{r['m']['staked']:.1f}", r), "right"),
                              ("Profit", lambda r: B(f"{r['m']['pnl']:+.1f}", r), "right"),
                              ("ROI", lambda r: B(fmt_pct(r["m"]["roi"]) if r["m"]["staked"] else "&ndash;", r), "right"),
                              ("Expected ROI", lambda r: B(fmt_pct(r["m"]["exp_roi"]), r), "right"),
                              ("Skill ROI", lambda r: B(fmt_pct(r["m"]["skill_roi"]), r), "right"),
                              ("Projected", lambda r: B(f"{r['m']['proj']:+.1f}", r), "right"),
                              ("Actual vs spread", lambda r: B(f"{r['m']['actual']:+.1f}", r), "right"),
                              ("Running total", lambda r: "" if r.get("total") else f"{r['cum']:+.1f}", "right"),
                              ("Shadow record", lambda r: B(r["s"]["rec"], r), "right"),
                              ("Shadow profit", lambda r: B(f"{r['s']['pnl']:+.1f}", r), "right"),
                              ("Shadow ROI", lambda r: B(fmt_pct(r["s"]["roi"]) if r["s"]["staked"] else "&ndash;", r), "right")]))

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
