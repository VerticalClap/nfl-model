# NFL Model (GitHub Automated)

This repo fetches schedule/odds/weather, trains a model, and publishes pick sheets automatically using GitHub Actions.

## Secrets required
- THE_ODDS_API_KEY: from the-odds-api.com
- NWS_USER_AGENT: e.g., "Stephen Soroka – stephen.soroka2424@yahoo.com"

## Local run (optional)
python scripts/run_all.py

Artifacts appear in ./cache/ or in GitHub Actions -> Artifacts.
