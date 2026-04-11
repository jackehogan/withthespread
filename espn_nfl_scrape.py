# nfl_scores_bs.py
import re, os, json, time, datetime as dt
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; scores-bot/1.0)"}

PFR_URL  = "https://www.pro-football-reference.com/years/{year}/week_{week}.htm"
ESPN_URL = "https://www.espn.com/nfl/scoreboard/_/week/{week}/year/{year}/seasontype/2"

def _get(url: str, tries: int = 3, timeout: int = 20) -> str:
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r.text
            last_err = RuntimeError(f"HTTP {r.status_code} for {url}")
        except Exception as e:
            last_err = e
        time.sleep(0.6)
    raise last_err

# -----------------------------
# Primary: Pro-Football-Reference
# -----------------------------
def _parse_pfr_week(html: str) -> List[Dict[str, Any]]:
    """
    Parse PFR 'week_X.htm' page.
    Returns list of dicts with: date (str or None), home_team, away_team, home_score, away_score, status="FINAL".
    """
    soup = BeautifulSoup(html, "html.parser")
    games_out: List[Dict[str, Any]] = []

    # Each game appears under a .game_summary container
    for gs in soup.select("div.game_summary"):
        # Pull date if present (PFR shows a small date in 'div.game_info' or 'table.game_info')
        game_date: Optional[str] = None
        info = gs.select_one(".game_info")
        if info:
            # look for something like "Sunday, January 5, 2025"
            txt = info.get_text(" ", strip=True)
            m = re.search(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}", txt)
            if m:
                try:
                    game_date = str(dt.datetime.strptime(m.group(0), "%A, %B %d, %Y").date())
                except Exception:
                    game_date = m.group(0)

        # The team table usually has two rows; we’ll read names and pts
        t = gs.select_one("table.teams")
        if not t:
            continue
        rows = [r for r in t.select("tr") if r.find("a")]
        if len(rows) < 2:
            continue

        def row_to_team_pts(tr) -> Optional[tuple]:
            # team name from the first <a>
            a = tr.find("a")
            if not a:
                return None
            team = a.get_text(strip=True)
            # points: look for an integer cell; on PFR it's typically a 'td' with digits
            pts = None
            for td in tr.find_all("td"):
                s = td.get_text(strip=True)
                if s.isdigit():
                    pts = int(s)
            return (team, pts)

        # PFR sometimes marks rows with classes 'winner'/'loser'/'draw'; we don't rely on that.
        pair = []
        for tr in rows:
            tp = row_to_team_pts(tr)
            if tp:
                pair.append(tp)
            if len(pair) == 2:
                break
        if len(pair) != 2:
            continue

        team1, pts1 = pair[0]
        team2, pts2 = pair[1]
        if pts1 is None or pts2 is None:
            # Sometimes the pts are missing if not final; skip those.
            continue

        # PFR does not guarantee home/away ordering here. Infer home/away by “@” marker if present in the row text.
        # Heuristic: in the second row, if you see '@' it's the away marker before opponent -> second row is away.
        # If ambiguous, just assign the higher score as winner but keep arbitrary home/away (stable order).
        txt_rows = [re.sub(r"\s+", " ", r.get_text(" ", strip=True)) for r in rows[:2]]
        # Example row text contains '@' when that row is away team.
        row1_has_at = " @ " in f" {txt_rows[0]} "
        row2_has_at = " @ " in f" {txt_rows[1]} "

        if row1_has_at and not row2_has_at:
            away_team, away_score = team1, pts1
            home_team, home_score = team2, pts2
        elif row2_has_at and not row1_has_at:
            away_team, away_score = team2, pts2
            home_team, home_score = team1, pts1
        else:
            # Fallback: keep as read but call first row away for consistency
            away_team, away_score = team1, pts1
            home_team, home_score = team2, pts2

        games_out.append({
            "date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "status": "FINAL"
        })

    return games_out

# -----------------------------
# Fallback: ESPN scoreboard HTML (__NEXT_DATA__ JSON)
# -----------------------------
def _deep_find_events(obj: Any) -> Optional[List[dict]]:
    """
    Recursively search for an 'events' list inside ESPN's Next.js JSON blob.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "events" and isinstance(v, list) and v and isinstance(v[0], dict):
                return v
            found = _deep_find_events(v)
            if found:
                return found
    elif isinstance(obj, list):
        for x in obj:
            found = _deep_find_events(x)
            if found:
                return found
    return None

def _parse_espn_week(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return []
    try:
        data = json.loads(script.string)
    except Exception:
        return []

    events = _deep_find_events(data) or []
    out: List[Dict[str, Any]] = []
    for ev in events:
        try:
            date_str = ev.get("date")
            comps = (ev.get("competitions") or [])[0]
            status = (((comps or {}).get("status") or {}).get("type") or {}).get("state", "")
            # Only finals
            if str(status).lower() not in {"final", "post", "postponed"} and "final" not in str(status).lower():
                continue

            teams = comps.get("competitors") or []
            # Map by homeAway
            home = next((t for t in teams if t.get("homeAway") == "home"), None)
            away = next((t for t in teams if t.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            out.append({
                "date": date_str.split("T")[0] if date_str else None,
                "home_team": home["team"]["displayName"],
                "away_team": away["team"]["displayName"],
                "home_score": int(float(home.get("score") or 0)),
                "away_score": int(float(away.get("score") or 0)),
                "status": "FINAL"
            })
        except Exception:
            continue

    return out

# -----------------------------
# Public function
# -----------------------------
def get_nfl_scores_bs(year: int, week: int) -> pd.DataFrame:
    """
    Scrape FINAL scores for a given NFL regular-season week/year using BeautifulSoup.
    Primary: Pro-Football-Reference. Fallback: ESPN scoreboard HTML.
    Returns a DataFrame with columns:
      ['date','home_team','away_team','home_score','away_score','status']
    """
    # Try PFR first
    try:
        pfr_html = _get(PFR_URL.format(year=year, week=week))
        games = _parse_pfr_week(pfr_html)
    except Exception:
        games = []

    if not games:
        # Fallback to ESPN (__NEXT_DATA__ inside the HTML)
        try:
            espn_html = _get(ESPN_URL.format(year=year, week=week))
            games = _parse_espn_week(espn_html)
        except Exception:
            games = []

    if not games:
        raise RuntimeError(f"No games found for week {week}, {year} via PFR/ESPN.")

    # Build tidy DataFrame
    df = pd.DataFrame(games)
    # Order columns and sort by date/home team for stability
    cols = ["date", "home_team", "away_team", "home_score", "away_score", "status"]
    for c in cols:
        if c not in df:
            df[c] = None
    df = df[cols].sort_values(["date", "home_team", "away_team"], na_position="last").reset_index(drop=True)
    return df

from datetime import date, timedelta


def nfl_week_sunday(year: int, week: int | None = None):
    """Return date objects for NFL regular-season Sundays.
       If week is given -> a single date; else -> list of all Sundays."""
    sept1 = date(year, 9, 1)
    labor = sept1 + timedelta(days=(7 - sept1.weekday()) % 7)  # first Monday in Sept
    week1_sun = labor + timedelta(days=6)                      # Sunday after Labor Day
    n_weeks = 18 if year >= 2021 else 17
    if week is not None:
        if not (1 <= week <= n_weeks): raise ValueError("week out of range")
        return week1_sun + timedelta(days=7*(week-1))
    return [week1_sun + timedelta(days=7*i) for i in range(n_weeks)]

# -----------------------------
# CLI example
# -----------------------------
if __name__ == "__main__":
    YEAR, WEEK = 2025, 1
    df_weeks = nfl_week_sunday(YEAR, WEEK)
    print(
        f"Week {WEEK} of {YEAR} is {df_weeks[0]} (Sunday) - {df_weeks[-1]} (Saturday)."
    )
    breakpoint()
    df = get_nfl_scores_bs(YEAR, WEEK)
    print(df.to_string(index=False))

