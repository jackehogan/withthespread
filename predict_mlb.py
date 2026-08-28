"""
Daily MLB prediction pipeline.

Fetches upcoming matchups, assembles all context features (starting pitcher,
bullpen, moneyline implied probability from run-line juice), trains on
2022-present history, and generates cover-probability predictions for the
run line (±1.5).

Usage
-----
    python predict_mlb.py                    # predict tomorrow's games
    python predict_mlb.py --days-ahead 2    # predict 2 days out
    python predict_mlb.py --season 2026     # explicit season year
    python predict_mlb.py --top 5           # show top N picks only
    python predict_mlb.py --max-evals 50    # larger hyperopt budget

Output
------
Predictions printed to stdout and upserted to MongoDB predictions collection.
Columns: team, opponent, spread (run line ±1.5), predspread, coverprob,
         ml_implied_prob, sp_name, home.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import smtplib
import time
from zoneinfo import ZoneInfo
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd

import db
import data_pipeline as dp
import model as ml
import reporting
from config import MLB


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def _read_email_config(config_path: str = "data/config.txt") -> dict | None:
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("email")
    except Exception:
        return None


def send_predictions_email(
    preds: pd.DataFrame,
    target_date: str,
    roi_summary: dict,
    history: pd.DataFrame | None = None,
    config_path: str = "data/config.txt",
) -> None:
    """
    Send today's top MLB picks + season ROI summary by email.

    `history` is the full predictions table; when supplied the email also carries
    yesterday's graded results and the nightly ROI chart.

    Requires an 'email' block in data/config.txt:
        {
          "smtp_host": "smtp.gmail.com",
          "smtp_port": 587,
          "username":  "you@gmail.com",
          "app_password": "xxxx xxxx xxxx xxxx",
          "to": "you@gmail.com"
        }

    For Gmail, generate an App Password at:
        myaccount.google.com → Security → 2-Step Verification → App passwords
    """
    cfg = _read_email_config(config_path)
    if not cfg:
        print("  Email config missing — skipping email.")
        return

    # ---- build HTML table — one compact row per game ----
    # Kelly sizing (0.10x fractional Kelly on a $100 bankroll) lives in
    # reporting.py so today's recommendation and the reconstruction of
    # yesterday's stakes cannot drift apart.
    _kelly_bet = reporting.kelly_bet

    # Order by recommended stake, largest first, so the biggest commitments are
    # read first. Kelly size is a function of both EV and price, so this is not
    # the same ordering as EV alone. Rows with no recommended bet sort last.
    _stakes = {
        team: (_kelly_bet(row.get("ev", float("nan")),
                          row.get("ml_odds", float("nan"))) or 0.0)
        for team, row in preds.iterrows()
    }
    preds = preds.loc[
        sorted(preds.index, key=lambda t: (-_stakes.get(t, 0.0), t))
    ]

    rows_html = []
    for team, row in preds.iterrows():
        ev = row.get("ev", float("nan"))
        # Moneyline bets: price is ml_odds and the probability shown is P(win).
        juice = row.get("ml_odds", float("nan"))
        bet_str = f"{team.split()[-1]} ML"
        prob_key   = "win_prob"
        juice_str  = f"{int(juice):+d}"    if pd.notna(juice) else "&#8212;"
        ev_str     = f"{ev:+.3f}"          if pd.notna(ev)    else "&#8212;"
        cover_str  = f"{row.get(prob_key, float('nan')):.1%}" if pd.notna(row.get(prob_key)) else "&#8212;"
        # A missing pitcher arrives as NaN, which is truthy — guard on notna,
        # or the whole email dies on .split() and the caller swallows it.
        sp = row.get("sp_name")
        sp = str(sp) if pd.notna(sp) and sp else ""
        sp_parts = sp.split() if sp else []
        sp_abbr = (f"{sp_parts[0][0]}.{' '.join(sp_parts[1:])}" if len(sp_parts) > 1 else sp)
        matchup = row.get("matchup") or f"vs {row.get('opponent','?')}"
        kelly_amt = _kelly_bet(ev, juice)
        bet_amt_str = f"${kelly_amt:.2f}" if kelly_amt is not None else "&#8212;"
        # Top SHAP reason — first item from the comma-separated reasons string
        reasons_raw = row.get("reasons")
        if pd.isna(reasons_raw) or not reasons_raw:
            top_reason = "&#8212;"
        else:
            top_reason = str(reasons_raw).split(",")[0].strip()
        game_time = row.get("game_time") or "&#8212;"
        highlight = ' style="background:#e8f5e9;"' if pd.notna(ev) and ev > 0 else ""
        rows_html.append(
            f"<tr{highlight}>"
            f"<td style='white-space:nowrap'>{game_time}</td>"
            f"<td>{matchup}</td><td><b>{bet_str}</b></td>"
            f"<td>{ev_str}</td><td>{juice_str}</td>"
            f"<td>{cover_str}</td><td style='font-weight:bold'>{bet_amt_str}</td>"
            f"<td>{sp_abbr}</td>"
            f"<td style='font-size:12px;color:#444;text-align:left'>{top_reason}</td>"
            f"</tr>"
        )

    table_html = (
        "<table class='picks'>"
        "<tr><th>Time</th><th>Matchup</th><th>Bet</th><th>EV</th><th>Line</th>"
        "<th>Prob%</th><th>Bet ($100 BR)</th><th>Pitcher</th><th>Top Reason</th></tr>"
        + "".join(rows_html)
        + "</table>"
    )

    # ---- yesterday's advised bets + the ROI tracker ----
    # Both are best-effort: a failure here must not cost the picks email.
    yday_html, chart_png = "", None
    if history is not None and not history.empty:
        try:
            yday_html = reporting.yesterday_html(
                reporting.yesterday_results(history, target_date), target_date)
        except Exception as exc:
            print(f"  Yesterday's results section failed: {exc}")
        try:
            chart_png = reporting.roi_chart_png(history)
        except Exception as exc:
            print(f"  ROI chart failed: {exc}")

    chart_html = (
        "<h3>Daily ROI by EV threshold</h3>"
        "<img src='cid:roichart' alt='Daily ROI by EV threshold' "
        "style='width:100%;max-width:700px;height:auto'/>"
        "<p style='color:#999;font-size:12px'>Dots are daily ROI, sized by bets "
        "placed that day. Line is the 7-day rolling mean, band is &plusmn;1 SE, "
        "dashed line is the average to date.</p>"
        if chart_png else ""
    )

    roi = roi_summary
    roi_line = (
        f"Season record: <b>{roi['wins']}-{roi['losses']}"
        + (f"-{roi['pushes']}" if roi['pushes'] else "")
        + f"</b> | Win rate: <b>{roi['win_rate']*100:.1f}%</b>"
        + f" | P&L: <b>{roi['total_pnl']:+.2f}u</b>"
        + f" | ROI: <b>{roi['roi_pct']:+.1f}%</b>"
        if roi["n_bets"] > 0
        else "No completed bets yet this season."
    )

    html = f"""
    <html><head><style>
      body      {{ font-family: Arial, sans-serif; font-size: 14px; }}
      h2        {{ color: #1a1a2e; }}
      table.picks {{ border-collapse: collapse; width: 100%; }}
      table.picks th {{ background: #1a1a2e; color: white; padding: 8px; }}
      table.picks td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: center; }}
      table.picks td.reasons {{ text-align: left; font-size: 12px; color: #555; min-width: 200px; }}
      table.picks tr:nth-child(even) {{ background: #f4f4f4; }}
      .roi {{ background: #eef7ee; padding: 10px; border-radius: 6px; margin-bottom: 16px; }}
    </style></head><body>
      <h2>⚾ MLB Run-Line Picks — {target_date}</h2>
      <div class="roi">{roi_line}</div>
      <h3>Today's picks (sorted by EV — positive EV = value bet)</h3>
      {table_html}
      {chart_html}
      {yday_html}
      <p style="color:#999; font-size:12px; margin-top:20px;">
        Generated by WithTheSpread · coverprob = model probability of covering +/-1.5 run line<br>
        Bet sizing: 0.10x fractional Kelly on $100 bankroll (stake = 100 x 0.10 x EV / payout).
        Scale linearly for other bankroll sizes (e.g. $1000 BR: multiply by 10).
        Only positive-EV bets shown a recommended size.
      </p>
    </body></html>
    """

    # ---- send ----
    # 'related' wraps the alternative part so the chart can be referenced by CID.
    # Inline PNG renders in Gmail; an <img> to an external URL would be blocked.
    msg = MIMEMultipart("related")
    msg["Subject"] = f"⚾ MLB Picks {target_date}"
    msg["From"]    = cfg["username"]
    msg["To"]      = cfg["to"]

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html, "html"))
    msg.attach(alt)

    if chart_png:
        img = MIMEImage(chart_png, _subtype="png")
        img.add_header("Content-ID", "<roichart>")
        img.add_header("Content-Disposition", "inline", filename="roi_chart.png")
        msg.attach(img)

    try:
        with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"])) as server:
            server.starttls()
            server.login(cfg["username"], cfg["app_password"])
            server.sendmail(cfg["username"], cfg["to"], msg.as_string())
        print(f"  Email sent to {cfg['to']}.")
    except Exception as exc:
        print(f"  Email failed: {exc}")


# ---------------------------------------------------------------------------
# Schedule + probable pitchers from statsapi (free, no key)
# ---------------------------------------------------------------------------

def fetch_upcoming_mlb_games(target_date: str) -> pd.DataFrame:
    """
    Fetch MLB schedule for a given date from statsapi.

    Parameters
    ----------
    target_date : 'YYYY-MM-DD'

    Returns
    -------
    DataFrame with columns: game_date, home_team, away_team, home_sp, away_sp
    """
    # The Eastern-date check below compares against this value as a string; a
    # date/Timestamp would make every comparison unequal and silently drop the
    # entire slate.
    target_date = str(target_date)[:10]

    try:
        import statsapi
    except ImportError:
        print("  MLB-StatsAPI not installed (pip install MLB-StatsAPI).")
        return pd.DataFrame()

    try:
        schedule = statsapi.schedule(
            start_date=target_date,
            end_date=target_date,
            sportId=1,
        )
    except Exception as exc:
        print(f"  statsapi.schedule failed: {exc}")
        return pd.DataFrame()

    rows = []
    for game in schedule:
        status = game.get("status", "")
        if status not in ("Scheduled", "Pre-Game", "Warmup", "Preview"):
            continue  # skip postponed / in-progress / final
        home = game.get("home_name", "")
        away = game.get("away_name", "")
        if not home or not away:
            continue
        # Filter to known MLB teams (excludes spring training / exhibition)
        if MLB.known_teams and (home not in MLB.known_teams or away not in MLB.known_teams):
            continue
        rows.append({
            "game_date": target_date,
            "home_team": home,
            "away_team": away,
            "home_sp":   game.get("home_probable_pitcher", "") or "",
            "away_sp":   game.get("away_probable_pitcher", "") or "",
            # UTC ISO; rendered in Eastern for display.
            "game_datetime": game.get("game_datetime", "") or "",
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Verify every game really starts on the requested Eastern date. statsapi
    # is asked for one date and returns that date's official games, but the
    # start times are UTC, so this confirms the two agree rather than assuming
    # it — the same mismatch that silently dropped West-Coast games elsewhere.
    if "game_datetime" in df.columns:
        et_dates = df["game_datetime"].apply(
            lambda s: _et_time_date(s) if s else target_date)
        off_day = df[et_dates != target_date]
        if not off_day.empty:
            for _, r in off_day.iterrows():
                print(f"  WARN: dropping {r['away_team']} @ {r['home_team']} — "
                      f"starts {et_dates[_]} ET, not {target_date}")
            df = df[et_dates == target_date]
        if df.empty:
            return pd.DataFrame()

    # Number games per matchup to flag doubleheaders
    df["game_num"] = df.groupby(["home_team", "away_team"]).cumcount() + 1
    return df


# ---------------------------------------------------------------------------
# Upcoming context assembly
# ---------------------------------------------------------------------------

def _et_time_date(iso_utc: str) -> str:
    """Eastern calendar date ('YYYY-MM-DD') of a UTC ISO timestamp."""
    if not iso_utc:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _et_time(iso_utc: str) -> str:
    """Render a UTC ISO timestamp as Eastern clock time, e.g. '7:05 PM ET'."""
    if not iso_utc:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    local = dt.astimezone(ZoneInfo("America/New_York"))
    return local.strftime("%-I:%M %p ET") if os.name != "nt" \
        else local.strftime("%I:%M %p ET").lstrip("0")


def _american_to_raw_prob(ml_val) -> float:
    """American odds → raw implied probability (includes vig)."""
    try:
        v = float(ml_val)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(v):
        return float("nan")
    return abs(v) / (abs(v) + 100.0) if v < 0 else 100.0 / (v + 100.0)


def build_upcoming_context_mlb(
    schedule_df: pd.DataFrame,
    spreads_df: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    bullpen_stats: pd.DataFrame,
    season_games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the upcoming_context DataFrame for today's MLB games.

    Parameters
    ----------
    schedule_df   : From fetch_upcoming_mlb_games().
    spreads_df    : From dp.fetch_upcoming_spreads(MLB, ...) — indexed by team.
    pitcher_stats : From dp.fetch_mlb_pitcher_stats(season-1) — indexed by name.
    bullpen_stats : From dp.fetch_mlb_bullpen_stats(season-1) — indexed by team.
    season_games  : Current-season completed games from MongoDB.

    Returns
    -------
    DataFrame indexed by team name with all context columns used by model.py.
    """
    if schedule_df.empty:
        return pd.DataFrame()

    # --- Same-day guard -------------------------------------------------
    # The odds board lists several days ahead, and odds are matched to teams by
    # NAME only. A team playing today and tomorrow could therefore be priced
    # off tomorrow's game. Restrict the board to the dates actually being
    # predicted before any lookup happens.
    target_dates = set(schedule_df["game_date"].astype(str).unique())
    if len(target_dates) > 1:
        print(f"  WARN: schedule spans multiple dates {sorted(target_dates)} — "
              f"expected a single day.")

    if not spreads_df.empty and "game_date" in spreads_df.columns:
        _before = len(spreads_df)
        spreads_df = spreads_df[spreads_df["game_date"].astype(str).isin(target_dates)]
        _dropped = _before - len(spreads_df)
        if _dropped:
            print(f"  Odds rows dropped for other dates: {_dropped} "
                  f"(kept {len(spreads_df)} for {sorted(target_dates)})")
    elif not spreads_df.empty:
        print("  WARN: odds have no game_date column — cannot verify they are "
              "for today. Prices may come from another day's game.")

    # Any team appearing twice on the board for these dates is a doubleheader;
    # the first occurrence is kept upstream, so flag it rather than silently
    # pricing game 2 off game 1.
    if not spreads_df.empty:
        _dupes = spreads_df.index[spreads_df.index.duplicated()].unique().tolist()
        if _dupes:
            print(f"  NOTE: {len(_dupes)} team(s) have multiple games on the "
                  f"board today (doubleheaders): {', '.join(map(str, _dupes[:5]))}")

    # Last played date per team — for is_b2b
    last_date: dict[str, str] = {}
    if not season_games.empty and "date" in season_games.columns:
        last_date = season_games.groupby("team")["date"].max().to_dict()

    rows: dict[str, dict] = {}  # keyed by team; last game wins for features (DH: game 1 odds used)

    # Count games per team to detect doubleheaders
    team_game_count: dict[str, int] = {}
    for _, game in schedule_df.iterrows():
        for t in (game["home_team"], game["away_team"]):
            team_game_count[t] = team_game_count.get(t, 0) + 1

    for _, game in schedule_df.iterrows():
        game_date  = game["game_date"]
        game_time  = _et_time(game.get("game_datetime", ""))
        game_num   = int(game.get("game_num", 1))
        home_team  = game["home_team"]
        away_team  = game["away_team"]
        dh_suffix  = f" (G{game_num})" if team_game_count.get(home_team, 1) > 1 else ""
        matchup_str = f"{away_team} @ {home_team}{dh_suffix}"

        matchups = [
            (home_team, away_team, game["home_sp"], 1),
            (away_team, home_team, game["away_sp"], 0),
        ]
        for team, opp, sp_name, is_home in matchups:
            if not team:
                continue

            # Back-to-back flag
            prev = last_date.get(team)
            is_b2b = 0.0
            if prev:
                gap = (pd.to_datetime(game_date) - pd.to_datetime(prev)).days
                is_b2b = 1.0 if gap == 1 else 0.0

            # Run line + juice + straight moneyline from the-odds-api
            spread = moneyline = ml_odds = over_under = np.nan
            if not spreads_df.empty and team in spreads_df.index:
                row = spreads_df.loc[team]
                spread     = row.get("spread",       np.nan)
                moneyline  = row.get("spread_juice", np.nan)  # run-line juice (spread EV)
                ml_odds    = row.get("moneyline",    np.nan)  # h2h win/lose odds (ML EV)
                over_under = row.get("total",        np.nan)

            # Starting pitcher stats (prior season).
            # pitcher_stats is indexed by name_key -- accent-stripped and
            # lowercased by dp._normalize_pitcher_name. Looking up the raw
            # schedule name matched NOTHING (0 of 28 starters), so sp_era,
            # sp_whip and sp_k9 were NaN for every prediction.
            sp_era = sp_whip = sp_k9 = sp_ip_per_start = np.nan
            _key = dp._normalize_pitcher_name(sp_name) if sp_name else ""
            if _key and not pitcher_stats.empty and _key in pitcher_stats.index:
                p = pitcher_stats.loc[_key]
                if isinstance(p, pd.DataFrame):
                    p = p.iloc[0]
                sp_era  = float(p["era"])  if pd.notna(p.get("era"))  else np.nan
                sp_whip = float(p["whip"]) if pd.notna(p.get("whip")) else np.nan
                sp_k9   = float(p["k9"])   if pd.notna(p.get("k9"))   else np.nan
                sp_ip_per_start = (float(p["ip_per_start"])
                                   if pd.notna(p.get("ip_per_start")) else np.nan)

            # Bullpen stats (prior season, team-level)
            bp_era = bp_whip = bp_k9 = bp_hr9 = bp_ip_per_game = np.nan
            if not bullpen_stats.empty and team in bullpen_stats.index:
                b = bullpen_stats.loc[team]
                bp_ip_per_game = (float(b["bp_ip_per_game"])
                                  if pd.notna(b.get("bp_ip_per_game")) else np.nan)
                bp_era  = float(b["bp_era"])  if pd.notna(b.get("bp_era"))  else np.nan
                bp_whip = float(b["bp_whip"]) if pd.notna(b.get("bp_whip")) else np.nan
                bp_k9   = float(b["bp_k9"])   if pd.notna(b.get("bp_k9"))   else np.nan
                bp_hr9  = float(b["bp_hr9"])  if pd.notna(b.get("bp_hr9"))  else np.nan

            rows[team] = {
                "opponent":    opp,
                "matchup":     matchup_str,
                "game_num":    game_num,
                "game_date":   game_date,
                "game_time":   game_time,
                "home":        float(is_home),
                "is_b2b":      is_b2b,
                "spread":      spread,
                "moneyline":   moneyline,
                "ml_odds":     ml_odds,
                "over_under":  over_under,
                "sp_name":     sp_name,
                "sp_era":      sp_era,
                "sp_whip":     sp_whip,
                "sp_k9":       sp_k9,
                "sp_ip_per_start": sp_ip_per_start,
                "bp_era":      bp_era,
                "bp_ip_per_game": bp_ip_per_game,
                "bp_whip":     bp_whip,
                "bp_k9":       bp_k9,
                "bp_hr9":      bp_hr9,
            }

    if not rows:
        return pd.DataFrame()

    ctx = pd.DataFrame(rows).T

    # --- Compute ml_implied_prob (no-vig) from both teams' run-line juice ---
    # Both sides are already in ctx["moneyline"] — look up opponent's price.
    if "moneyline" in ctx.columns and "opponent" in ctx.columns:
        p_self = ctx["moneyline"].apply(_american_to_raw_prob)
        p_opp  = ctx["opponent"].map(lambda opp: _american_to_raw_prob(ctx.loc[opp, "moneyline"])
                                      if opp in ctx.index else np.nan)
        total  = p_self + p_opp
        ctx["ml_implied_prob"] = (p_self / total).where(total > 0)
    else:
        ctx["ml_implied_prob"] = np.nan

    # --- Pitcher + bullpen matchup edges ---
    if "opponent" in ctx.columns:
        def _opp(team: str, col: str):
            opp = ctx.loc[team, "opponent"] if team in ctx.index else None
            if opp and opp in ctx.index:
                return ctx.loc[opp, col]
            return np.nan

        if "sp_era" in ctx.columns:
            ctx["opp_sp_era"]  = [_opp(t, "sp_era")  for t in ctx.index]
            ctx["opp_sp_whip"] = [_opp(t, "sp_whip") for t in ctx.index]
            ctx["sp_era_edge"] = ctx["opp_sp_era"].astype(float) - ctx["sp_era"].astype(float)

        if "bp_era" in ctx.columns:
            ctx["opp_bp_era"]  = [_opp(t, "bp_era")  for t in ctx.index]
            ctx["opp_bp_whip"] = [_opp(t, "bp_whip") for t in ctx.index]
            ctx["bp_era_edge"] = ctx["opp_bp_era"].astype(float) - ctx["bp_era"].astype(float)

    # --- Own-side features computed from completed games ---------------------
    # model.py's _compute_context builds these for TRAINING. This function is a
    # separate implementation used only at PREDICTION time, and the two had
    # drifted: 22 of 46 model features arrived as all-NaN, which XGBoost happily
    # accepts, so the failure was silent and produced nonsense probabilities
    # (mean win_prob 0.395 against a training base rate of 0.500). Anything
    # added to _compute_context must be mirrored here until the duplication is
    # removed. The fill guard in _run is the backstop.
    ctx = _attach_history_features(ctx, season_games)

    # --- Opponent mirrors ----------------------------------------------------
    # Both sides of every game are rows in ctx, so mapping through `opponent`
    # gives the opponent's value for THIS game -- the same game_pk keying the
    # training path uses.
    if "opponent" in ctx.columns:
        # Drive off the live matchup set so this cannot drift when features
        # move between _OPP_FEATURES and the bullpen switch.
        # The switchable set, plus the mirrors that live in _matchup_cols and
        # are therefore always expected regardless of _MATCHUP_MODE.
        _want = set(ml._matchup_extra_features()) | {
            "opp_sp_era_rolling", "opp_sp_era", "opp_sp_whip",
            "opp_bp_era", "opp_bp_whip"}
        _MIRROR = [c for c in ctx.columns if ("opp_" + c) in _want]
        for _c in _MIRROR:
            tgt = "opp_" + _c
            if tgt in ctx.columns:
                continue
            ctx[tgt] = ctx["opponent"].map(
                lambda o, col=_c: ctx.loc[o, col] if o in ctx.index else np.nan)

    return ctx


def _attach_history_features(ctx: pd.DataFrame,
                             season_games: pd.DataFrame) -> pd.DataFrame:
    """
    Form and rolling-bullpen features for the upcoming game, from completed
    games only. Mirrors what model._compute_context builds for training.
    """
    if season_games is None or season_games.empty:
        return ctx
    g = season_games.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    for c in ("diff", "bp_ip_game", "bp_er_game", "bp_bb_game",
              "bp_h_game", "bp_k_game", "bp_pitch_game"):
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.sort_values(["team", "date"])

    # form: last game, mean of last 5, current win/loss streak
    form = {}
    for team, grp in g.dropna(subset=["diff"]).groupby("team"):
        d = grp["diff"].to_numpy()
        w = l = 0
        for v in d[::-1]:
            if v > 0 and l == 0: w += 1
            elif v < 0 and w == 0: l += 1
            else: break
        form[team] = {"1_ago_diff": d[-1] if len(d) else np.nan,
                      "diff_mean_5": d[-5:].mean() if len(d) else np.nan,
                      "win_streak": float(w), "loss_streak": float(l)}
    for col in ("1_ago_diff", "diff_mean_5", "win_streak", "loss_streak"):
        ctx[col] = [form.get(t, {}).get(col, np.nan) for t in ctx.index]

    # bullpen: 30-day pitch load, and lambda-shrunk in-season rates
    if "bp_pitch_game" in g.columns:
        cutoff = g["date"].max() - pd.Timedelta(days=ml._BP_WORKLOAD_DAYS)
        load = g[g["date"] > cutoff].groupby("team")["bp_pitch_game"].sum().to_dict()
        ctx["bp_pitch_30d"] = [load.get(t, np.nan) for t in ctx.index]

    _SPEC = {"era": (["bp_er_game"], 9.0, "bp_era"),
             "whip": (["bp_bb_game", "bp_h_game"], 1.0, "bp_whip"),
             "k9": (["bp_k_game"], 9.0, "bp_k9")}
    # sp_era_rolling: this starter's own in-season ERA shrunk toward his
    # prior-season number, the same lambda blend model.py uses for training.
    if {"sp_name", "sp_ip_game", "sp_er_game"} <= set(g.columns):
        g["sp_ip_game"] = pd.to_numeric(g["sp_ip_game"], errors="coerce")
        g["sp_er_game"] = pd.to_numeric(g["sp_er_game"], errors="coerce")
        agg = (g.dropna(subset=["sp_name"])
                 .groupby("sp_name")[["sp_ip_game", "sp_er_game"]].sum())
        lam = ml._SP_ERA_BLEND_LAMBDA
        vals = {}
        for team in ctx.index:
            nm = ctx.loc[team, "sp_name"] if "sp_name" in ctx.columns else None
            prior = ctx.loc[team, "sp_era"] if "sp_era" in ctx.columns else np.nan
            prior = float(prior) if pd.notna(prior) else np.nan
            std = np.nan
            if nm and nm in agg.index:
                ip = float(agg.loc[nm, "sp_ip_game"])
                er = float(agg.loc[nm, "sp_er_game"])
                if ip > 0:
                    std = er * 9.0 / ip
            if not np.isfinite(std):     vals[team] = prior
            elif not np.isfinite(prior): vals[team] = std
            else:
                w = float(agg.loc[nm, "sp_ip_game"])
                w = w / (w + lam)
                vals[team] = w * std + (1 - w) * prior
        ctx["sp_era_rolling"] = [vals.get(t, np.nan) for t in ctx.index]

    for rate, (nums, mult, prior_col) in _SPEC.items():
        if not set(nums) <= set(g.columns) or "bp_ip_game" not in g.columns:
            continue
        lam = ml._BP_RATE_LAMBDA[rate]
        vals = {}
        for team, grp in g.groupby("team"):
            ip = grp["bp_ip_game"].sum(skipna=True)
            num = grp[nums].sum(axis=1, skipna=False).sum(skipna=True)
            std = (num * mult / ip) if ip > 0 else np.nan
            prior = ctx.loc[team, prior_col] if (team in ctx.index
                    and prior_col in ctx.columns) else np.nan
            prior = float(prior) if pd.notna(prior) else np.nan
            if not np.isfinite(std):      vals[team] = prior
            elif not np.isfinite(prior):  vals[team] = std
            else:
                wgt = ip / (ip + lam)
                vals[team] = wgt * std + (1 - wgt) * prior
        ctx[f"bp_{rate}_rolling"] = [vals.get(t, np.nan) for t in ctx.index]
    return ctx


# ---------------------------------------------------------------------------
# Main prediction run
# ---------------------------------------------------------------------------

def run_mlb(
    season: int,
    target_date: str,
    max_evals: int = 20,
    top_n: int | None = None,
    key_type: str = "free",
) -> pd.DataFrame:
    """
    Full daily MLB prediction pipeline.

    Parameters
    ----------
    season      : Current MLB season year (e.g. 2026).
    target_date : 'YYYY-MM-DD' — the date of games to predict.
    max_evals   : Hyperopt budget for model tuning.
    top_n       : If set, return only the top N predictions by coverprob.
    key_type    : 'free' or 'paid' for the-odds-api.com.

    Returns
    -------
    DataFrame of predictions for today's games, sorted by coverprob descending.
    """
    print(f"\n{'=' * 60}")
    print(f"MLB Predictions — {target_date}  (season {season})")
    print(f"{'=' * 60}")

    client = db.connect()
    try:
        return _run(client, season, target_date, max_evals, top_n, key_type)
    finally:
        client.close()


_FEATURE_LABELS = {
    # Rolling SpreadScore lags
    "ss_lag_1":       "last game SS",
    "ss_lag_2":       "2-game SS",
    "ss_lag_3":       "3-game SS",
    "ss_lag_5":       "5-game SS",
    "ss_lag_7":       "7-game SS",
    "ss_lag_10":      "10-game SS",
    "ss_lag_15":      "15-game SS",
    "ss_mean":        "avg SS",
    "ss_std":         "SS consistency",
    "ss_trend":       "SS trend",
    # Market / context
    "ml_implied_prob":"market prob",
    "home":           "home field",
    "is_b2b":         "back-to-back",
    # Pitcher
    "sp_era":         "SP ERA",
    "sp_whip":        "SP WHIP",
    "sp_k9":          "SP K/9",
    "opp_sp_era":     "opp SP ERA",
    "opp_sp_whip":    "opp SP WHIP",
    "sp_era_edge":    "SP ERA edge",
    # Bullpen
    "bp_era":         "bullpen ERA",
    "bp_whip":        "bullpen WHIP",
    "bp_era_edge":    "bullpen ERA edge",
    "bp_ip_14d":      "bullpen fatigue",
    # Elo
    "elo_diff":       "Elo edge",
    "opponent_elo":   "opp Elo",
    # Style
    "style_edge":     "style matchup",
}


def _shap_reasons(clf, X_pred: pd.DataFrame, teams, top_n: int = 3) -> pd.Series:
    """
    Compute SHAP values for X_pred and return top-N driving features per team
    as a human-readable string, e.g. "SP ERA edge (+0.8), Elo edge (+0.5), market prob (-0.3)".
    """
    try:
        import shap
        feat_cols = clf.get_booster().feature_names
        X = X_pred.set_index("team")[feat_cols] if "team" in X_pred.columns else X_pred[feat_cols]
        explainer  = shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(X)   # shape (n_teams, n_features)
        shap_df    = pd.DataFrame(shap_vals, index=X.index, columns=feat_cols)

        reasons = {}
        for team in teams:
            if team not in shap_df.index:
                reasons[team] = ""
                continue
            row = shap_df.loc[team].abs().nlargest(top_n)
            parts = []
            for feat in row.index:
                val   = shap_df.loc[team, feat]
                label = _FEATURE_LABELS.get(feat, feat)
                parts.append(f"{label} ({val:+.2f})")
            reasons[team] = ", ".join(parts)
        return pd.Series(reasons)
    except Exception:
        return pd.Series("", index=teams)


def _run(
    client,
    season: int,
    target_date: str,
    max_evals: int,
    top_n: int | None,
    key_type: str,
) -> pd.DataFrame:

    # ------------------------------------------------------------------
    # 0. Data freshness gate — must pass before anything else runs
    # ------------------------------------------------------------------
    # For a forward prediction (target_date >= today), the DB must contain
    # games through yesterday. For a retroactive prediction, through target_date - 1.
    # If data is stale, auto-seed and re-check. Abort if still behind.
    _today      = pd.Timestamp.today().normalize()
    _target_dt  = pd.to_datetime(target_date)
    _required_through = min(_target_dt - pd.Timedelta(days=1), _today - pd.Timedelta(days=1))

    _all_raw = db.fetch_games(client, "mlb")
    _season_raw = _all_raw[_all_raw["season"] == season] if not _all_raw.empty else pd.DataFrame()

    if not _season_raw.empty and "date" in _season_raw.columns:
        _most_recent = pd.to_datetime(_season_raw["date"]).max()
        _days_stale  = (_required_through - _most_recent).days
    else:
        _most_recent = pd.Timestamp("2000-01-01")
        _days_stale  = 999

    if _days_stale > 0:
        if _days_stale > 1:
            print(f"  ERROR: DB is {_days_stale} days stale "
                  f"(last={_most_recent.date()}, need={_required_through.date()}). "
                  f"Run manually to fill the full gap:\n"
                  f"    python seed_mlb.py --seasons {season} --since {_most_recent.date()}")
            return pd.DataFrame()

        print(f"[0/5] DB last seeded {_most_recent.date()} — need data through "
              f"{_required_through.date()} (1d gap). Seeding now...")
        import subprocess, sys as _sys
        subprocess.run(
            [_sys.executable, "seed_mlb.py", "--seasons", str(season),
             "--since", str(_most_recent.date())],
        )

        # Re-check after seed
        _all_raw    = db.fetch_games(client, "mlb")
        _season_raw = _all_raw[_all_raw["season"] == season] if not _all_raw.empty else pd.DataFrame()
        if not _season_raw.empty and "date" in _season_raw.columns:
            _most_recent = pd.to_datetime(_season_raw["date"]).max()
            _days_stale  = (_required_through - _most_recent).days

        if _days_stale > 0:
            print(f"  ERROR: DB still {_days_stale}d stale after seeding "
                  f"(last={_most_recent.date()}, need={_required_through.date()}). "
                  f"Aborting — predictions on incomplete data are unreliable.")
            return pd.DataFrame()

    # Odds coverage check on training data.
    # spreadscore = diff + spread, so NaN spread → NaN target → row dropped from training.
    # If a large share of recent games are missing spread data the model is learning
    # from an unrepresentative sample and EV can't be computed at inference time.
    _ODDS_LOOKBACK_GAMES = 50   # check last N games per team
    _ODDS_MIN_FILL       = 0.80 # abort if spread fill rate below this
    _recent_season = _season_raw.copy() if not _season_raw.empty else pd.DataFrame()
    if not _recent_season.empty and "spread" in _recent_season.columns:
        _recent_n   = min(_ODDS_LOOKBACK_GAMES, len(_recent_season))
        _recent_g   = _recent_season.nlargest(_recent_n, "period") if "period" in _recent_season.columns else _recent_season.tail(_recent_n)
        _spread_fill = _recent_g["spread"].notna().mean()
        # Check the run-line price the model actually consumes. spread_juice is
        # the correctly-labelled column; `moneyline` is the legacy overloaded
        # one and is empty on rows seeded since the split, so checking it alone
        # made this gate abort on freshly-seeded data.
        if "spread_juice" in _recent_g.columns:
            _price = _recent_g["spread_juice"]
            if "moneyline" in _recent_g.columns:
                _price = _price.fillna(_recent_g["moneyline"])
        elif "moneyline" in _recent_g.columns:
            _price = _recent_g["moneyline"]
        else:
            _price = pd.Series(dtype=float)
        _ml_fill = _price.notna().mean() if len(_price) else 0.0
        if _spread_fill < _ODDS_MIN_FILL or _ml_fill < _ODDS_MIN_FILL:
            print(f"  ERROR: Odds coverage too low on last {_recent_n} game rows — "
                  f"spread={_spread_fill:.0%}, run-line price={_ml_fill:.0%} "
                  f"(minimum {_ODDS_MIN_FILL:.0%}). "
                  f"Run seed_mlb.py to backfill odds before predicting.")
            return pd.DataFrame()
        print(f"[0/5] Data current through {_most_recent.date()}. "
              f"Odds coverage OK (spread={_spread_fill:.0%}, price={_ml_fill:.0%}).")
    else:
        print(f"[0/5] Data current through {_most_recent.date()}. OK.")

    # Per-team row count check — detect teams with significantly fewer games
    # than the median (indicates missing data for specific teams).
    if not _season_raw.empty and "team" in _season_raw.columns:
        _counts = _season_raw.groupby("team")["period"].count()
        _median_count = _counts.median()
        _lagging = _counts[_counts < _median_count * 0.85]
        if not _lagging.empty:
            print(f"  WARNING: The following teams have significantly fewer games "
                  f"than expected (median={_median_count:.0f}):")
            for _t, _c in _lagging.items():
                print(f"    {_t}: {_c} games")
            print(f"  Run: python seed_mlb.py --seasons {season} --since {_most_recent.date()}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1. Today's schedule + probable pitchers
    # ------------------------------------------------------------------
    print(f"[1/5] Fetching schedule for {target_date}...")
    schedule_df = fetch_upcoming_mlb_games(target_date)
    if schedule_df.empty:
        # Fallback: pull schedule from DB for past dates (games already "Final")
        _hist = db.fetch_games(client, "mlb")
        _hist = _hist[_hist["date"] == target_date].dropna(subset=["opponent"])
        if not _hist.empty:
            print(f"  Live schedule empty — reconstructing from DB ({len(_hist)} rows).")
            _rows = []
            seen = set()
            for _, r in _hist.iterrows():
                key = tuple(sorted([r["team"], r["opponent"]]))
                if key in seen:
                    continue
                seen.add(key)
                home = r["team"] if r.get("home", 0) == 1 else r["opponent"]
                away = r["opponent"] if r.get("home", 0) == 1 else r["team"]
                _rows.append({
                    "game_date":  target_date,
                    "home_team":  home,
                    "away_team":  away,
                    "home_sp":    "",
                    "away_sp":    "",
                    "game_num":   1,
                })
            schedule_df = pd.DataFrame(_rows)
        else:
            print("  No games found.")
            return pd.DataFrame()
    print(f"  {len(schedule_df)} games — "
          f"{', '.join(f'{r.away_team} @ {r.home_team}' for _, r in schedule_df.iterrows())}")

    # Warn on missing probable pitchers
    n_missing = sum(
        1 for _, r in schedule_df.iterrows()
        if not r["home_sp"] or not r["away_sp"]
    )
    if n_missing:
        print(f"  WARN: {n_missing} game(s) missing at least one probable pitcher "
              f"— pitcher features will be NaN for those slots.")

    # ------------------------------------------------------------------
    # 2. Run-line odds from the-odds-api
    # ------------------------------------------------------------------
    print(f"[2/5] Fetching run lines from the-odds-api ({key_type})...")
    _is_past = pd.to_datetime(target_date).date() < pd.Timestamp.today().normalize().date()
    spreads_df = pd.DataFrame()
    if not _is_past:
        try:
            spreads_df = dp.fetch_upcoming_spreads(MLB, key_type=key_type)
            if spreads_df.empty:
                print("  No odds returned — spread/moneyline features will be NaN.")
            else:
                n_matched = sum(
                    1 for _, r in schedule_df.iterrows()
                    if r["home_team"] in spreads_df.index or r["away_team"] in spreads_df.index
                )
                print(f"  Odds found for {n_matched}/{len(schedule_df)} games.")
        except Exception as exc:
            print(f"  Odds fetch failed: {exc}")
    else:
        # Past date: reconstruct odds from stored DB values
        _hist_odds = db.fetch_games(client, "mlb")
        _hist_odds = _hist_odds[_hist_odds["date"] == target_date].dropna(subset=["opponent"])
        if not _hist_odds.empty:
            _odds_rows = {}
            for _, r in _hist_odds.iterrows():
                _odds_rows[r["team"]] = {
                    "spread":      r.get("spread",    np.nan),
                    "spread_juice":r.get("moneyline", np.nan),
                    "moneyline":   r.get("ml_odds",   np.nan),
                    "total":       np.nan,
                }
            spreads_df = pd.DataFrame(_odds_rows).T
            n_matched = spreads_df["spread"].notna().sum()
            print(f"  Past date — odds from DB: {n_matched}/{len(spreads_df)} with spread data.")
        else:
            print("  Past date — no stored odds found, features will be NaN.")

    # ------------------------------------------------------------------
    # 3. Prior-season pitcher + bullpen stats
    # ------------------------------------------------------------------
    prior_season = season - 1
    print(f"[3/5] Loading {prior_season} pitcher & bullpen stats...")
    pitcher_stats = dp.fetch_mlb_pitcher_stats(prior_season)
    time.sleep(0.3)
    bullpen_stats = dp.fetch_mlb_bullpen_stats(prior_season)

    if pitcher_stats.empty:
        print("  WARN: No pitcher stats returned.")
    else:
        print(f"  {len(pitcher_stats)} starters loaded.")

    if bullpen_stats.empty:
        print("  WARN: No bullpen stats returned.")
    else:
        print(f"  {len(bullpen_stats)} team bullpens loaded.")

    # ------------------------------------------------------------------
    # 4. Load saved model + build features
    # ------------------------------------------------------------------
    # Load all historical games (2022–present) from MongoDB
    all_games = db.fetch_games(client, "mlb")
    if all_games.empty:
        print("  No historical games in DB — run seed_mlb.py first.")
        return pd.DataFrame()

    # Apply regular-season date filter per season
    seasons_present = all_games["season"].unique()
    all_games = pd.concat(
        [dp.filter_regular_season(all_games[all_games["season"] == s], MLB, s)
         for s in seasons_present],
        ignore_index=True,
    )

    # Current-season games only (for rolling features and context)
    season_games = all_games[all_games["season"] == season].copy()

    # next_period = one past the highest completed game number for playing teams.
    # Use the median across teams playing today so the training cutoff is stable.
    playing_teams = set(schedule_df["home_team"]) | set(schedule_df["away_team"])
    if not season_games.empty:
        periods_per_team = (
            season_games[season_games["team"].isin(playing_teams)]
            .groupby("team")["period"].max()
        )
        next_period = int(periods_per_team.median()) + 1 if not periods_per_team.empty else 1
    else:
        next_period = 1  # opening day — no prior games

    print(f"  Next period (median game #): {next_period}")

    # ------------------------------------------------------------------
    # 4. Load saved model — weights are fixed; only features update nightly
    # ------------------------------------------------------------------
    # Hyperparameters (lookback, Elo K, bp_days) and XGBoost weights come
    # from the saved pkl. Run _kelly_analysis.py to retrain when new seasons
    # complete. The style_model is always rebuilt from current season data
    # because it captures this season's head-to-head matchup edges.
    print("[4/5] Loading saved model...")
    try:
        _bundle      = ml.load_model()
        clf          = _bundle["clf"]
        scores       = _bundle["scores"]
        best_lookback= _bundle["best_lookback"]
        best_k       = _bundle["best_k"]
    except FileNotFoundError:
        print("  ERROR: No saved model found at data/mlb_model.pkl.")
        print("  Run _kelly_analysis.py to train and save the model first.")
        return pd.DataFrame()

    # The StyleModel used to be refit here on every run — a full SVD of the
    # season's matchup matrix — purely to produce style_edge, which the model
    # does not use (see model._USE_STYLE_EDGE). Skipped; nothing downstream
    # needs it, and build_prediction_features accepts None.
    style_model = None
    print(f"  lookback={best_lookback}, Elo K={best_k}, bp_ip_14d={ml._BP_FATIGUE_DAYS}d (fixed) | "
          f"val ROC={scores.get('clf_val_roc', float('nan')):.3f}  "
          f"val acc={scores.get('clf_val_acc', float('nan')):.1%}")

    # Assemble upcoming context
    upcoming_context = build_upcoming_context_mlb(
        schedule_df, spreads_df, pitcher_stats, bullpen_stats, season_games
    )

    # Rolling bullpen fatigue for upcoming games using the CV-selected window.
    # Uses a date-based sum (not period-based) so it works correctly for the
    # upcoming game, which has no period entry in the DB yet.
    if not upcoming_context.empty and "bp_ip_game" in season_games.columns:
        fat_map = ml.compute_bp_fatigue_for_date(season_games, target_date, ml._BP_FATIGUE_DAYS)
        if fat_map:
            for team in upcoming_context.index:
                val = fat_map.get(team, np.nan)
                upcoming_context.loc[team, "bp_ip_14d"] = val
            n_filled = upcoming_context["bp_ip_14d"].notna().sum()
            print(f"  bp_ip_14d (fatigue, {ml._BP_FATIGUE_DAYS}d) populated for "
                  f"{n_filled}/{len(upcoming_context)} teams.")

    # bp_ip_14d and is_b2b are filled after the context is assembled, so their
    # opponent mirrors have to be taken here rather than inside the builder.
    if not upcoming_context.empty and "opponent" in upcoming_context.columns:
        for _c in ("bp_ip_14d", "is_b2b"):
            if _c in upcoming_context.columns:
                upcoming_context["opp_" + _c] = upcoming_context["opponent"].map(
                    lambda o, col=_c: upcoming_context.loc[o, col]
                    if o in upcoming_context.index else np.nan)

    # ------------------------------------------------------------------
    # 5. Predict
    # ------------------------------------------------------------------
    print("[5/5] Generating predictions...")

    X_pred = ml.build_prediction_features(
        season_games,
        next_period,
        best_lookback,
        season,
        upcoming_context if not upcoming_context.empty else None,
        style_model,
        best_k,
    )

    if X_pred.empty:
        print("  Not enough season data to generate predictions.")
        return pd.DataFrame()

    # --- Feature fill guard --------------------------------------------------
    # Training and prediction build features through two separate code paths
    # (model._compute_context vs build_upcoming_context_mlb). When they drift, a
    # feature the model was trained on arrives as all-NaN. XGBoost accepts NaN
    # silently, so the run "succeeds" and emits nonsense -- that is exactly how
    # 22 of 46 features went missing and produced a mean win_prob of 0.395
    # against a 0.500 base rate, with one pick priced at +0.80 EV.
    #
    # Fail loudly instead. A feature the model actually splits on must not be
    # mostly empty at prediction time.
    _wc = _bundle.get("win_clf")
    if _wc is not None:
        _need = list(_wc.get_booster().feature_names)
        _empty, _sparse = [], []
        for _f in _need:
            if _f not in X_pred.columns:
                _empty.append(_f); continue
            _fill = pd.to_numeric(X_pred[_f], errors="coerce").notna().mean()
            if _fill == 0.0:
                _empty.append(_f)
            elif _fill < 0.5:
                _sparse.append((_f, _fill))
        if _sparse:
            print("  WARN: sparse features at prediction time: "
                  + ", ".join(f"{f} {v:.0%}" for f, v in _sparse[:8]))
        if _empty:
            raise RuntimeError(
                f"{len(_empty)} of {len(_need)} model features are EMPTY at "
                f"prediction time: {_empty[:10]}"
                + (" ..." if len(_empty) > 10 else "")
                + ". The training and prediction feature paths have drifted -- "
                  "anything added to model._compute_context must also be built "
                  "in build_upcoming_context_mlb. Refusing to predict."
            )

    preds = ml.predict(clf, X_pred, win_clf=_bundle.get("win_clf"))

    # Attach context columns for display
    display_cols = ["opponent", "matchup", "game_num", "game_time", "home", "spread",
                    "moneyline", "ml_odds", "ml_implied_prob", "sp_name"]
    if not upcoming_context.empty:
        for col in display_cols:
            if col in upcoming_context.columns:
                preds[col] = preds.index.map(upcoming_context[col])

    # ------------------------------------------------------------------
    # Enforce p(A) + p(B) = 1 across the two sides of a game
    # ------------------------------------------------------------------
    # The model scores each team independently and nothing constrains it to
    # treat one game consistently from both sides. Measured over 2313 held-out
    # 2025 games the two win probabilities summed to between 0.609 and 1.366,
    # with 54% of games off by more than 0.05 -- that incoherence is what
    # produced moneyline EVs above +1.00.
    #
    # Splitting the discrepancy evenly projects onto the space where the pair
    # is coherent, and it is strictly better on every metric over those games:
    #   AUC      0.6969 -> 0.7193
    #   log loss 0.6370 -> 0.6321
    #   Brier    0.2230 -> 0.2205
    # Rows with p > 0.70 fall from 235 to 163, which is the inflated tail.
    if "opponent" in preds.columns and "win_prob" in preds.columns:
        if preds.index.has_duplicates:
            # Doubleheaders would mis-pair a team with the wrong game.
            print("  WARN: duplicate team rows -- skipping win_prob symmetrisation.")
        else:
            _wp  = pd.to_numeric(preds["win_prob"], errors="coerce")
            _sym = _wp.copy()
            _n   = 0
            for _t in preds.index:
                _o = preds.at[_t, "opponent"]
                if pd.isna(_o) or _o not in _wp.index:
                    continue
                if pd.isna(_wp[_t]) or pd.isna(_wp[_o]):
                    continue
                _sym[_t] = _wp[_t] + (1.0 - (_wp[_t] + _wp[_o])) / 2.0
                _n += 1
            if _n:
                preds["win_prob_raw"] = _wp
                preds["win_prob"] = _sym.clip(1e-6, 1.0 - 1e-6)
                print(f"  Symmetrised win_prob for {_n} team rows "
                      f"(largest shift {(_sym - _wp).abs().max():.3f}).")

    # Relative coverprob diff vs opponent (useful for edge signal display)
    def _rel(row, col):
        opp = row.get("opponent")
        if pd.isna(opp) or opp not in preds.index:
            return np.nan
        return float(row[col]) - float(preds.loc[opp, col])

    preds["coverprob_diff"] = preds.apply(_rel, col="coverprob", axis=1)

    # Model edge over market (coverprob - ml_implied_prob)
    if "ml_implied_prob" in preds.columns:
        preds["edge"] = preds["coverprob"] - preds["ml_implied_prob"].astype(float)
    else:
        preds["edge"] = preds["coverprob"] - 0.5

    def _calc_ev(prob: float, odds: float) -> float:
        """EV per $1 staked given win probability and American odds."""
        try:
            if np.isnan(prob) or np.isnan(odds):
                return np.nan
            payout = 100.0 / abs(odds) if odds < 0 else odds / 100.0
            return prob * payout - (1.0 - prob)
        except (TypeError, ValueError):
            return np.nan

    # Price each market against its own probability and its own odds, then take
    # whichever is better.
    #   SPREAD : coverprob vs the run-line juice
    #   ML     : win_prob  vs the head-to-head moneyline
    # These were previously conflated — cover probability was priced against
    # whatever sat in `moneyline`, which for ESPN-seeded rows was the h2h price.
    preds["ev_spread"] = preds.apply(
        lambda r: _calc_ev(float(r["coverprob"]), float(r.get("moneyline", np.nan))),
        axis=1,
    )
    preds["ev_ml"] = preds.apply(
        lambda r: _calc_ev(float(r.get("win_prob", np.nan)),
                           float(r.get("ml_odds", np.nan))),
        axis=1,
    )

    # Bet the moneyline only. Backtested out of sample on 2026 (market-blind
    # model), ROI at the 0.05 threshold:
    #
    #     run line only     5.4%
    #     moneyline only   10.3%
    #     max of the two    5.3%
    #
    # Taking max(ev_spread, ev_ml) never beat moneyline-only at any threshold.
    # With two noisy estimates the maximum is biased upward, so the rule finds
    # whichever market the model is most wrong about in the favourable
    # direction rather than the one where it has an edge.
    #
    # The reason the moneyline is the better market is not that the model
    # predicts winning better -- it predicts both about equally (AUC .5698
    # covering, .5704 winning). It is that the book prices the run line far
    # better than it prices the moneyline (.5996 vs .5650), because the run
    # line needs a model of the MARGIN distribution and the book has one.
    # Our team-strength signal transfers fully to the moneyline and only
    # partly to the run line.
    #
    # ev_spread is still computed and stored for comparison.
    preds["bet"] = "ML"
    preds["ev"] = preds["ev_ml"]

    # Drop teams with no game scheduled today (no opponent in context)
    preds = preds[preds["opponent"].notna()]

    # We bet the moneyline, so ml_odds is what EV needs. The run-line price is
    # no longer sufficient. (The model itself needs no price at all now — see
    # model._MARKET_BLIND — so this is purely about being able to price a bet.)
    no_odds = preds[preds["ml_odds"].isna()]
    if not no_odds.empty:
        print(f"  Dropped {len(no_odds)} team(s) with no moneyline price "
              f"(no EV computable): {', '.join(no_odds.index.tolist())}")
    preds = preds[preds["ml_odds"].notna()]

    if preds.empty:
        print("  No actionable predictions — all games missing odds data.")
        return pd.DataFrame()

    preds = preds.sort_values("ev", ascending=False)

    if top_n:
        preds = preds.head(top_n)

    # SHAP top-3 reasons per team, explained against the model that actually
    # drove the bet. Since the moneyline restructure that is win_clf; using the
    # cover model here described why a team should beat the run line next to a
    # bet on it winning outright, which are different questions (ats_elo_diff
    # is rank 2 of 27 by gain in the cover model and rank 12 of 26 in the win
    # model). Falls back to clf for bundles saved without a win model.
    _reason_clf = _bundle.get("win_clf") if str(preds["bet"].iloc[0]) == "ML" else clf
    preds["reasons"] = _shap_reasons(_reason_clf or clf, X_pred, preds.index)

    # Upsert to MongoDB
    db.upsert_predictions(client, [
        {
            "sport": "mlb", "team": team, "season": season, "period": next_period,
            "opponent":        row.get("opponent") if pd.notna(row.get("opponent")) else None,
            "spread":          float(row["spread"])          if pd.notna(row.get("spread"))          else None,
            "moneyline":       float(row["moneyline"])       if pd.notna(row.get("moneyline"))       else None,
            "ml_implied_prob": float(row["ml_implied_prob"]) if pd.notna(row.get("ml_implied_prob")) else None,
            "sp_name":         row.get("sp_name") or None,
            "coverprob":       float(row["coverprob"]) if pd.notna(row.get("coverprob")) else None,
            "win_prob":        float(row["win_prob"])  if pd.notna(row.get("win_prob"))        else None,
            "coverprob_diff":  row["coverprob_diff"]  if pd.notna(row.get("coverprob_diff"))  else None,
            "edge":            float(row["edge"])      if pd.notna(row.get("edge"))            else None,
            "ev":              float(row["ev"])        if pd.notna(row.get("ev"))              else None,
            "ev_spread":       float(row["ev_spread"]) if pd.notna(row.get("ev_spread"))       else None,
            "ev_ml":           float(row["ev_ml"])     if pd.notna(row.get("ev_ml"))           else None,
            "bet":             row.get("bet"),
            "ml_odds":         float(row["ml_odds"])   if pd.notna(row.get("ml_odds"))         else None,
            "reasons":         row.get("reasons") or None,
            "prediction_date": target_date,
        }
        for team, row in preds.iterrows()
    ])
    print(f"  Stored {len(preds)} predictions.")

    # -- Resolve any pending results from previous predictions --
    n_resolved = db.update_prediction_results(client, "mlb", season)
    if n_resolved:
        print(f"  Resolved results for {n_resolved} previous prediction(s).")

    # -- ROI summary --
    roi = db.fetch_roi_summary(client, "mlb", season)
    if roi["n_bets"] > 0:
        print(f"  Season ROI: {roi['wins']}-{roi['losses']} "
              f"({roi['win_rate']*100:.1f}% win rate), "
              f"P&L {roi['total_pnl']:+.2f}u, ROI {roi['roi_pct']:+.1f}%")

    # -- Send email --
    # Fetched after update_prediction_results above so yesterday's games are
    # already graded by the time the results table is built.
    try:
        history = db.fetch_predictions(client, "mlb")
    except Exception as exc:
        print(f"  Could not load prediction history: {exc}")
        history = None

    try:
        send_predictions_email(preds, target_date, roi, history=history)
    except Exception as exc:
        print(f"  Email failed (non-fatal): {exc}")

    # -- Auto-bet via ProphetX (only if configured and enabled) --
    import prophetx as px_mod
    if px_mod.is_enabled():
        try:
            px = px_mod.ProphetXClient.from_config()
            print(f"  [ProphetX] Fetching run-line markets for {target_date}...")
            markets = px.get_mlb_run_lines(target_date)
            print(f"  [ProphetX] {len(markets)} team markets found.")
            bet_results = px.place_model_bets(preds, markets, target_date)
            if not bet_results.empty:
                mode    = "DRY_RUN" if px.dry_run else "PLACED"
                actioned = bet_results[bet_results["status"] == mode]
                skipped  = bet_results[bet_results["status"] != mode]
                if not actioned.empty:
                    label = "would place" if px.dry_run else "placed"
                    env   = "sandbox" if px.sandbox else "PRODUCTION"
                    print(f"  [ProphetX/{env}] {len(actioned)} bet(s) {label} "
                          f"(${actioned['stake'].sum():.2f} total):")
                    for _, r in actioned.iterrows():
                        print(f"    {r['team']:<28} {r['bet']:<6}  "
                              f"odds={int(r['snapped_odds']):+d}  stake=${r['stake']:.2f}  "
                              f"EV={r['ev']:+.3f} (model {r['model_ev']:+.3f})  "
                              f"id={r['wager_id']}")
                if not skipped.empty:
                    print(f"  [ProphetX] {len(skipped)} selection(s) not actioned:")
                    for _, r in skipped.iterrows():
                        print(f"    {r['team']:<28} {r['status']}: {r['error']}")
            else:
                print("  [ProphetX] No bets placed (no markets matched or below min_ev).")
        except Exception as exc:
            print(f"  [ProphetX] ERROR: {exc}")

    return preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily MLB run-line prediction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--season", type=int,
        default=datetime.date.today().year,
        help="MLB season year (default: current year).",
    )
    parser.add_argument(
        "--days-ahead", type=int, default=0, dest="days_ahead",
        help="How many days ahead to predict (default: 0 = today).",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Explicit target date YYYY-MM-DD (overrides --days-ahead).",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Show only the top N picks by coverprob.",
    )
    parser.add_argument(
        "--max-evals", type=int, default=20, dest="max_evals",
        help="Hyperopt evaluation budget (default: 20).",
    )
    parser.add_argument(
        "--key-type", choices=["free", "paid"], default="free", dest="key_type",
        help="Odds API key type (default: free).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.date:
        target = args.date
    else:
        target = (datetime.date.today() + datetime.timedelta(days=args.days_ahead)).isoformat()

    result = run_mlb(
        season=args.season,
        target_date=target,
        max_evals=args.max_evals,
        top_n=args.top,
        key_type=args.key_type,
    )

    if not result.empty:
        _ABB = {
            "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
            "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
            "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
            "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
            "Colorado Rockies": "COL", "Detroit Tigers": "DET",
            "Houston Astros": "HOU", "Kansas City Royals": "KC",
            "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
            "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
            "Minnesota Twins": "MIN", "New York Mets": "NYM",
            "New York Yankees": "NYY", "Athletics": "OAK",
            "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
            "San Diego Padres": "SD", "San Francisco Giants": "SF",
            "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
            "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
            "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
        }

        def _abb(name): return _ABB.get(name, name.split()[-1][:3].upper())

        print("\n" + "=" * 68)
        print(f"  MLB PREDICTIONS -- {target}")
        print("=" * 68)
        print(f"  {'Time':<10} {'Matchup':<18} {'Bet':<12} {'EV':>6} {'Line':>6} {'Prob%':>7}  Pitcher")
        print("-" * 68)
        for team, row in result.iterrows():
            ev = row.get("ev", float("nan"))
            dh = " G2" if row.get("game_num", 1) == 2 else "   "
            # Moneyline bets: price and label come from ml_odds, not the run line.
            juice = row.get("ml_odds", float("nan"))
            bet_str = f"{_abb(team)} ML"
            opp = row.get("opponent") or "?"
            matchup = f"{_abb(opp)}@{_abb(team)}{dh}" if row.get("home", 0) == 1.0 else f"{_abb(team)}@{_abb(opp)}{dh}"
            # Show the probability for the market being bet: P(win), not P(cover).
            prob_key  = "win_prob"
            juice_str = f"{int(juice):+d}" if pd.notna(juice) else "  --"
            ev_str    = f"{ev:+.3f}"       if pd.notna(ev)    else "    --"
            cover_str = f"{row.get(prob_key, float('nan')):.1%}" if pd.notna(row.get(prob_key)) else "  --"
            sp = row.get("sp_name") or ""
            sp_parts = sp.split() if sp else []
            sp_abbr = (f"{sp_parts[0][0]}.{' '.join(sp_parts[1:])}" if len(sp_parts) > 1 else sp)[:14]
            marker = " *" if pd.notna(ev) and ev > 0 else ""
            gt = (row.get("game_time") or "")[:10]
            print(f"  {gt:<10} {matchup:<18} {bet_str:<12} {ev_str:>6} {juice_str:>6} {cover_str:>7}  {sp_abbr}{marker}")
