import pandas as pd

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal placeholder: pick common columns if present
    cols = [
        "off_epa","def_epa","off_success","def_success","off_pass_rate",
        "is_home","rest_days","travel_miles",
        "temp_f","wind_mph","gust_mph","precip_prob",
        "qb_out","ol_starters_out","db_starters_out"
    ]
    use = [c for c in cols if c in df.columns]
    return df[use].fillna(0.0)
