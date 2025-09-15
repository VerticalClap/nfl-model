import os

# Central place for environment variables and defaults
DATA_CACHE_DIR = os.environ.get("DATA_CACHE_DIR", "./cache")

# Odds API (optional, may not be set)
THE_ODDS_API_KEY = os.environ.get("THE_ODDS_API_KEY")

# NWS user agent (optional, but required for weather endpoints)
NWS_USER_AGENT = os.environ.get("NWS_USER_AGENT", "default@example.com")
