import pandas as pd

def extract_consensus_spreads(raw, books=[]):
    """
    Extract consensus point spreads from odds API response.
    Returns DataFrame with home/away spread lines and probabilities.
    """
    rows = []
    for ev in raw:
        home = ev["home_team"]
        away = ev["away_team"]
        bookmakers = ev.get("bookmakers", [])
        spreads = []
        for bk in bookmakers:
            if books and bk["key"] not in books:
                continue
            for mkt in bk.get("markets", []):
                if mkt["key"] == "spreads":
                    outcomes = {o["name"]: o for o in mkt["outcomes"]}
                    if home in outcomes and away in outcomes:
                        spreads.append((
                            bk["key"],
                            outcomes[home]["point"],
                            outcomes[away]["point"],
                            outcomes[home]["price"],
                            outcomes[away]["price"],
                        ))
        if spreads:
            df = pd.DataFrame(spreads, columns=["book", "home_spread", "away_spread", "home_price", "away_price"])
            # consensus = median line across books
            rows.append({
                "home_team": home,
                "away_team": away,
                "home_spread": df["home_spread"].median(),
                "away_spread": df["away_spread"].median(),
                "home_spread_price": df["home_price"].median(),
                "away_spread_price": df["away_price"].median(),
            })
    return pd.DataFrame(rows)
