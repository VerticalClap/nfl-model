import pandas as pd

def basic_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Add very simple example features to the schedule.
    Later, you can expand with team stats, weather, ELO, etc.
    """
    df = schedule.copy()
    df["is_home"] = df["home_team"] == df["team"]
    df["week_mod"] = df["week"] % 4  # example engineered feature
    return df
