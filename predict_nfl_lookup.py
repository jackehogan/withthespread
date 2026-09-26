"""
Score one NFL week with the 4-input lookup model.

For each team, find past team-games (1999..last season, both sides of every game) whose
four inputs look like this one - our last-game spread and spreadscore, the opponent's
last-game spread and spreadscore - weighted by a Gaussian kernel whose width is a multiple of
each input's standard deviation (std from the training seasons): 0.6 sd for the spreads (they
matter only roughly), 0.45 sd for the spreadscores. Take their cover rate. No shrinkage.
No hand-made weights or parts. Stake = 1 unit per point above break-even, cap 8;
only bets of 0.6+ units (cutoff sweep: 0-1.0 all give about the same profit). Week 18 not bet.

Inputs: data/nflverse_games.csv (results, closing lines), FanDuel Tuesday line
and price from data/nfl_odds_snapshots.parquet (consensus at -110 if missing).

    python predict_nfl_lookup.py --season 2026 --week 3
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, required=True)
ap.add_argument("--week", type=int, required=True)
ap.add_argument("--start", type=int, default=1999, help="first training season (default 1999)")
ap.add_argument("--c-spread", type=float, default=0.6, help="spread smoothing width, in standard deviations (default 0.6)")
ap.add_argument("--c-ss", type=float, default=0.45, help="spreadscore smoothing width, in standard deviations (default 0.45)")
ap.add_argument("--min-stake", type=float, default=0.6, help="smallest bet placed, units (default 0.6)")
ap.add_argument("--sort", choices=["kickoff", "cover"], default="kickoff", help="order games by kickoff or by P(cover)")
ap.add_argument("--no-shadow", action="store_true", help="skip the shadow model (n-2 correction)")
ap.add_argument("--json", help="also write the bets (main + shadow) to this JSON file")
a = ap.parse_args()
TRAIN = list(range(a.start, a.season))
SHADOW_SCALE = 0.5
if a.week >= 18:
    sys.exit("week 18 is not bet (teams rest starters)")

g = pd.read_csv("data/nflverse_games.csv")
g = g[(g.game_type == "REG") & g.season.between(a.start, a.season)].dropna(subset=["spread_line"])
g = g.sort_values(["season", "gameday", "gametime", "game_id"]).reset_index(drop=True)
o = pd.read_parquet("data/nfl_odds_snapshots.parquet")
o = o[(o.role == "tuesday") & (o.market == "spreads")]
fd = o[o.book == "fanduel"].groupby("game_id")[["home_point", "home_price", "away_price"]].first()
g["tue"] = g.game_id.map(fd.home_point).fillna(g.game_id.map(o.groupby("game_id").home_point.median()))
g["hp"], g["ap"] = g.game_id.map(fd.home_price), g.game_id.map(fd.away_price)

side = lambda tm, op, s: pd.DataFrame({
    "game_id": g.game_id, "season": g.season, "week": g.week, "gidx": g.index, "gameday": g.gameday,
    "gametime": g.gametime, "team": g[tm], "opp": g[op], "margin": s * g.result, "exp": s * g.spread_line,
    "tue": s * g.tue, "price": g.hp if s > 0 else g.ap, "home": s > 0})
t = (pd.concat([side("home_team", "away_team", 1), side("away_team", "home_team", -1)])
     .sort_values(["season", "team", "gidx"]).reset_index(drop=True))
gr = t.groupby(["season", "team"])
t["ss1"] = gr.margin.shift(1) - gr.exp.shift(1)
t["line1"] = gr.exp.shift(1)
t["m1"], t["opp1"] = gr.margin.shift(1), gr.opp.shift(1)
t = t.join(t.set_index(["game_id", "team"])[["ss1", "line1"]].add_prefix("opp_"), on=["game_id", "opp"])
t["r"] = np.sign(t.margin - t.exp)
hist = t[t.season.isin(TRAIN) & t.ss1.notna() & t.opp_ss1.notna() & t.r.notna()].reset_index(drop=True)
now = t[(t.season == a.season) & (t.week == a.week)].reset_index(drop=True)
if now.ss1.isna().any():
    sys.exit("some teams have no previous game yet this season - results not in")

# the lookup: kernel-weighted cover rate of past team-games with similar four inputs
COLS = ["line1", "ss1", "opp_line1", "opp_ss1"]
tr = hist[hist.r != 0]
Zt, Zq = tr[COLS].to_numpy(float), now[COLS].to_numpy(float)
WID = np.array([a.c_spread, a.c_ss, a.c_spread, a.c_ss]) * Zt.std(0)
W = np.exp(-0.5 * (((Zq[:, None, :] - Zt[None, :, :]) / WID) ** 2).sum(2))
p = 0.5 * (1 + (W @ tr.r.to_numpy(float)) / W.sum(1))
now = now.assign(similar=W.sum(1))

line = now.tue.where(now.tue.notna(), now.exp)
price = np.where(now.tue.notna(), now.price.fillna(-110.0), -110.0)
pay = np.where(price < 0, 100 / -price, price / 100)
be = 1 / (1 + pay)
b = now.assign(P=p, line_bet=line, price_bet=price, be=be, stake=np.clip((p - be) * 100, 0, 8))
# one row per game: the side with the larger stake, else the side the model leans to
b = b.sort_values(["stake", "P"], ascending=False).drop_duplicates("game_id")
b = b.sort_values("P", ascending=False) if a.sort == "cover" else b.sort_values(["gameday", "gametime", "game_id"])

print(f"\n{a.season} week {a.week} | 4-input lookup, trained on {TRAIN[0]}-{TRAIN[-1]} ({len(hist):,} team-games), "
      f"widths: spread {a.c_spread:g} sd = {WID[0]:.1f} pts, spreadscore {a.c_ss:g} sd = {WID[1]:.1f} pts")
print(f"stake = 1 unit per point above break-even, cap 8; bets placed at {a.min_stake:g}+ units "
      f"(stakes in brackets are below that and not bet)\n")
opp_info = t.set_index(["game_id", "team"])[["m1", "opp1", "line1"]]
last = lambda tm, m, op, ln, ss: (f"{tm} {'won' if m > 0 else 'lost'} by {abs(int(m))} vs {op} "
                                  f"as {'fav' if ln > 0 else 'dog'} {abs(ln):g} (ss {ss:+g})")
reason = lambda tm, m, op, ln, ss: (f"{tm} {'won' if m > 0 else 'lost'} by {abs(int(m))} vs {op} as "
                                    f"{'fav' if ln > 0 else 'dog'} {abs(ln):g}, "
                                    + (f"beat spread by {ss:g}" if ss > 0 else (f"missed spread by {-ss:g}" if ss < 0 else "push")))
# 2020-25 walk-forward: bets with under 25 similar games went 64-26 (+161 u); 25+ was about break-even
RARITY = lambda n: "RARE (<25)" if n < 25 else ("uncommon (25-100)" if n < 100 else "common (100+)")


def graded(margin, line_bet, stake, price):
    """(WIN/LOSS/PUSH or None if not played, units won)"""
    if np.isnan(margin):
        return None, 0.0
    cm_ = margin - line_bet
    pay_ = 100 / -price if price < 0 else price / 100
    return ("WIN", stake * pay_) if cm_ > 0 else (("LOSS", -stake) if cm_ < 0 else ("PUSH", 0.0))


rows, slip, main_json = [], [], []
for r in b.itertuples():
    om, oo, ol = opp_info.loc[(r.game_id, r.opp)]
    is_bet = r.stake >= a.min_stake
    pick = f"{r.team} {-r.line_bet:+g}"
    status = "" if np.isnan(r.margin) else (f"{'WIN' if r.margin - r.line_bet > 0 else ('PUSH' if r.margin == r.line_bet else 'LOSS')}"
                                             f" ({r.team} {'won' if r.margin > 0 else 'lost'} by {abs(int(r.margin))})")
    if is_bet:
        slip.append({"bet on": pick, "matchup": f"{r.opp} @ {r.team}" if r.home else f"{r.team} @ {r.opp}",
                     "price": int(r.price_bet), "units": f"{r.stake:.1f}", "P(cover)": f"{100*r.P:.1f}%",
                     "break-even": f"{100*r.be:.1f}%", "similar games": f"{r.similar:.0f}",
                     "situation": RARITY(r.similar),
                     "reason: bet team last game": reason(r.team, r.m1, r.opp1, r.line1, r.ss1),
                     "reason: opponent last game": reason(r.opp, om, oo, ol, r.opp_ss1), "result": status})
        res_j, pnl_j = graded(r.margin, r.line_bet, r.stake, r.price_bet)
        main_json.append({"bet": pick, "matchup": slip[-1]["matchup"], "kickoff": f"{r.gameday} {r.gametime}",
                          "price": int(r.price_bet), "units": round(float(r.stake), 2), "p_cover": round(float(r.P), 4),
                          "break_even": round(float(r.be), 4), "similar": round(float(r.similar), 1),
                          "situation": RARITY(r.similar), "reason_team": slip[-1]["reason: bet team last game"],
                          "reason_opp": slip[-1]["reason: opponent last game"], "result": res_j, "pnl": round(pnl_j, 2)})
    rows.append({"matchup": f"{r.opp} @ {r.team}" if r.home else f"{r.team} @ {r.opp}",
                 "BET ON": f"{pick} ({r.stake:.1f}u)" if is_bet else "-",
                 "model leans": pick, "price": int(r.price_bet), "P(cover)": f"{100*r.P:.1f}%",
                 "break-even": f"{100*r.be:.1f}%", "stake": f"{r.stake:.1f}" if is_bet else f"({r.stake:.1f})",
                 "similar": f"{r.similar:.0f}",
                 "lean team last game": last(r.team, r.m1, r.opp1, r.line1, r.ss1),
                 "opponent last game": last(r.opp, om, oo, ol, r.opp_ss1),
                 "result": status})
print("BET SLIP")
print(pd.DataFrame(slip).to_string(index=False) if slip else "  no bets this week")
print("  (similar games = past team-games like this one; 2020-25 at 1+ units: RARE bets went 64-26, +161 u; the rest ~break-even)")

print()
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 60)
print(pd.DataFrame(rows).to_string(index=False))
nb = b[b.stake >= a.min_stake]
print(f"\nbets placed: {len(nb)}, total stake {nb.stake.sum():.1f} units")
if len(nb) and nb.margin.notna().any():
    s = nb[nb.margin.notna()]
    cm = s.margin - s.line_bet
    pnl = np.select([cm > 0, cm < 0], [s.stake * np.where(s.price_bet < 0, 100 / -s.price_bet, s.price_bet / 100), -s.stake], 0.0)
    print(f"graded so far: {int((cm > 0).sum())}-{int((cm < 0).sum())}, {pnl.sum():+.1f} units")

# ---------------------------------------------------------------------------------------------
# SHADOW MODEL (tracked, not bet): correct last game's line and spreadscore using game n-2.
# The lookup's prediction for each team's previous game (walk-forward: season u from seasons < u)
# says how wrong that game's line was likely to be; x 0.5, in points (sd_ats x inverse-normal(P)).
# corrected line n-1 = line n-1 + adj, corrected spreadscore n-1 = spreadscore n-1 - adj, both teams.
# Walk-forward 2020-25: +215 u vs +194 u for the main model, but the gain faded in 2024-25.
if not a.no_shadow:
    from scipy.stats import norm

    def kernel(train, query, cols):
        trr = train[train.r != 0]
        zt, zq = trr[cols].to_numpy(float), query[cols].to_numpy(float)
        wid = np.array([a.c_spread, a.c_ss, a.c_spread, a.c_ss]) * zt.std(0)
        w_ = np.exp(-0.5 * (((zq[:, None, :] - zt[None, :, :]) / wid) ** 2).sum(2))
        return 0.5 * (1 + (w_ @ trr.r.to_numpy(float)) / w_.sum(1)), w_.sum(1)

    valid = t.ss1.notna() & t.opp_ss1.notna()
    P_game = pd.Series(0.5, index=t.index)
    for u in range(a.start + 1, a.season + 1):
        q = t[valid & (t.season == u) & ((t.week < a.week) | (u < a.season))]
        if len(q):
            P_game.loc[q.index] = kernel(t[valid & (t.season < u) & t.r.notna()], q, COLS)[0]
    sd_ats = (t.margin - t.exp)[t.season.isin(TRAIN)].std()
    t["adj_out"] = SHADOW_SCALE * sd_ats * norm.ppf(P_game.clip(0.01, 0.99))
    t["adj_in"] = t.groupby(["season", "team"]).adj_out.shift(1).fillna(0.0)
    t = t.drop(columns=[c for c in ["opp_adj_in"] if c in t]).join(
        t.set_index(["game_id", "team"])[["adj_in"]].rename(columns={"adj_in": "opp_adj_in"}), on=["game_id", "opp"])
    t["opp_adj_in"] = t.opp_adj_in.fillna(0.0)
    t["line1c"], t["ss1c"] = t.line1 + t.adj_in, t.ss1 - t.adj_in
    t["opp_line1c"], t["opp_ss1c"] = t.opp_line1 + t.opp_adj_in, t.opp_ss1 - t.opp_adj_in
    CC = ["line1c", "ss1c", "opp_line1c", "opp_ss1c"]
    hist_c = t[t.season.isin(TRAIN) & valid & t.r.notna()]
    now_c = t[(t.season == a.season) & (t.week == a.week)].reset_index(drop=True)
    p_c, n_c = kernel(hist_c, now_c, CC)
    line_c = now_c.tue.where(now_c.tue.notna(), now_c.exp)
    price_c = np.where(now_c.tue.notna(), now_c.price.fillna(-110.0), -110.0)
    pay_c = np.where(price_c < 0, 100 / -price_c, price_c / 100)
    be_c = 1 / (1 + pay_c)
    sb = now_c.assign(P=p_c, similar=n_c, line_bet=line_c, price_bet=price_c, pay=pay_c, be=be_c,
                      stake=np.clip((p_c - be_c) * 100, 0, 8))
    sb = sb.sort_values(["stake", "P"], ascending=False).drop_duplicates("game_id")
    sb = sb[sb.stake >= a.min_stake]
    main_bets = {f"{r.team} {-r.line_bet:+g}" for r in b[b.stake >= a.min_stake].itertuples()}
    print(f"\nSHADOW (not bet): n-2 correction x {SHADOW_SCALE:g}")
    rows_s, shadow_json = [], []
    for r in sb.itertuples():
        pick = f"{r.team} {-r.line_bet:+g}"
        res_ = "" if np.isnan(r.margin) else ("WIN" if r.margin - r.line_bet > 0 else ("PUSH" if r.margin == r.line_bet else "LOSS"))
        rows_s.append({"bet on": pick, "price": int(r.price_bet), "units": f"{r.stake:.1f}", "P(cover)": f"{100*r.P:.1f}%",
                       "similar games": f"{r.similar:.0f}", "last-game corrections (us / opp)": f"{r.adj_in:+.1f} / {r.opp_adj_in:+.1f} pts",
                       "also main bet?": "yes" if pick in main_bets else "NO", "result": res_})
        res_j, pnl_j = graded(r.margin, r.line_bet, r.stake, r.price_bet)
        shadow_json.append({"bet": pick, "matchup": f"{r.opp} @ {r.team}" if r.home else f"{r.team} @ {r.opp}",
                            "kickoff": f"{r.gameday} {r.gametime}", "price": int(r.price_bet),
                            "units": round(float(r.stake), 2), "p_cover": round(float(r.P), 4),
                            "similar": round(float(r.similar), 1), "also_main": pick in main_bets,
                            "corrections": f"{r.adj_in:+.1f} / {r.opp_adj_in:+.1f} pts", "result": res_j, "pnl": round(pnl_j, 2)})
    print(pd.DataFrame(rows_s).to_string(index=False) if rows_s else "  no bets this week")
    if len(sb) and sb.margin.notna().any():
        s_ = sb[sb.margin.notna()]
        cm_ = s_.margin - s_.line_bet
        pnl_ = np.select([cm_ > 0, cm_ < 0], [s_.stake * s_.pay, -s_.stake], 0.0)
        print(f"shadow graded: {int((cm_ > 0).sum())}-{int((cm_ < 0).sum())}, {pnl_.sum():+.1f} units")

if a.json:
    import json
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump({"season": a.season, "week": a.week, "train": f"{TRAIN[0]}-{TRAIN[-1]}",
                   "widths": f"spread {a.c_spread:g} sd, spreadscore {a.c_ss:g} sd", "min_stake": a.min_stake,
                   "main": main_json, "shadow": None if a.no_shadow else shadow_json}, fh, indent=1)
