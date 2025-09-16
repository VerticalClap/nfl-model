# nfl_model/odds.py
import pandas as pd

TEAM_MAP = {"LA":"LAR","SD":"LAC","OAK":"LV"}  # normalize

def _norm_team(x: str) -> str:
    if not isinstance(x, str):
        return x
    return TEAM_MAP.get(x, x)

def _vig_fair(p_home_raw: float, p_away_raw: float) -> tuple[float, float]:
    # simple normalization (sum to 1)
    s = (p_home_raw or 0) + (p_away_raw or 0)
    if s > 0:
        return (p_home_raw or 0)/s, (p_away_raw or 0)/s
    return None, None

def _ml_to_prob(ml: float | None) -> float | None:
    if ml is None:
        return None
    try:
        ml = float(ml)
    except:
        return None
    if ml > 0:
        return 100.0/(ml+100.0)
    else:
        return (-ml)/( -ml + 100.0)

def extract_consensus_moneylines(raw: list, books: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for ev in raw:
        home = ev.get("home_team")
        away = ev.get("away_team")
        bks = ev.get("bookmakers", [])
        if books:
            bks = [b for b in bks if b.get("key") in books]
        ml_home, ml_away = [], []
        for bk in bks:
            for m in bk.get("markets", []):
                if m.get("key") == "h2h":
                    out = m.get("outcomes", [])
                    for o in out:
                        if o.get("name") == home:
                            ml_home.append(o.get("price"))
                        elif o.get("name") == away:
                            ml_away.append(o.get("price"))
        if not ml_home or not ml_away:
            continue
        h = sum(ml_home)/len(ml_home)
        a = sum(ml_away)/len(ml_away)
        p_h_raw = _ml_to_prob(h)
        p_a_raw = _ml_to_prob(a)
        p_h, p_a = _vig_fair(p_h_raw, p_a_raw)
        rows.append(dict(
            home_team=_norm_team(home), away_team=_norm_team(away),
            home_ml=h, away_ml=a, home_prob=p_h, away_prob=p_a,
            home_prob_raw=p_h_raw, away_prob_raw=p_a_raw,
        ))
    return pd.DataFrame(rows)

def extract_consensus_spreads(raw: list, books: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for ev in raw:
        home = ev.get("home_team")
        away = ev.get("away_team")
        bks = ev.get("bookmakers", [])
        if books:
            bks = [b for b in bks if b.get("key") in books]

        lines, home_odds, away_odds = [], [], []
        for bk in bks:
            for m in bk.get("markets", []):
                if m.get("key") != "spreads":
                    continue
                for o in m.get("outcomes", []):
                    if o.get("name") == home:
                        if o.get("point") is not None:
                            lines.append(float(o.get("point")))
                        if o.get("price") is not None:
                            home_odds.append(float(o.get("price")))
                    elif o.get("name") == away:
                        if o.get("price") is not None:
                            away_odds.append(float(o.get("price")))
        if not lines:
            continue
        rows.append(dict(
            home_team=_norm_team(home), away_team=_norm_team(away),
            home_line=sum(lines)/len(lines),
            home_spread_odds=sum(home_odds)/len(home_odds) if home_odds else None,
            away_spread_odds=sum(away_odds)/len(away_odds) if away_odds else None,
        ))
    return pd.DataFrame(rows)
