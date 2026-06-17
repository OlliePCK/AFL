# AFL Prediction Model

An end-to-end machine learning system that predicts Australian Football League (AFL) match outcomes and margins, compares its forecasts against live market odds, and serves the results through an interactive web dashboard.

The project covers the full ML lifecycle — automated data collection, feature engineering, model training and tuning, prediction serving, backtesting, and containerised deployment with scheduled retraining.

> **Disclaimer:** This is a personal project built for learning and research into sports modelling and probability. It is not financial or betting advice.

## Overview

The system ingests AFL data going back to 2012 from multiple public sources, engineers a feature set per match, and trains gradient-boosted models (CatBoost) to predict both win probability and margin. A calibration layer compares the model's probabilities against live market odds to gauge how well-calibrated the forecasts are. Everything is surfaced in a Next.js dashboard and runs on a schedule inside Docker.

## Features

- **Automated data collection** from the Squiggle API, AFLTables and FootyWire, covering results, fixtures, player data and team selections (2012–present).
- **Feature engineering** combining team form, player availability, venue and weather signals (weather pulled from the Visual Crossing API).
- **CatBoost models** for match outcome and margin, with **Optuna** hyperparameter and feature optimisation and ensemble/stacking experiments.
- **Probability calibration** evaluated against live market odds to measure forecast quality.
- **Live prediction pipeline** that updates the dataset with the latest results and generates predictions each round.
- **Walk-forward backtesting framework** (validated across 2019–2025 seasons) for evaluating model and strategy variants.
- **Interactive dashboard** (Next.js + Recharts) presenting predictions and model performance.
- **Containerised + scheduled** via Docker and cron for hands-off data collection and retraining.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Language | Python 3 |
| ML / Data | CatBoost, scikit-learn, Optuna, pandas, NumPy |
| Scraping | requests, BeautifulSoup, lxml |
| External APIs | The Odds API (market odds), Visual Crossing (weather), Anthropic (match summaries) |
| Dashboard | Next.js 16, React 19, TypeScript, Tailwind CSS, shadcn/ui, Recharts |
| Infra | Docker, docker-compose, cron; self-hosted on Unraid via GitHub Actions → Docker Hub |

## Project Structure

```
AFL/
├── src/                    # Core library
│   ├── squiggle_client.py        # Squiggle API client
│   ├── afltables_scraper.py      # AFLTables scraper
│   ├── footywire_scraper.py      # FootyWire scraper
│   ├── selection_scraper.py      # Team selection / availability
│   ├── dataset_builder.py        # Builds the master training dataset
│   ├── features.py               # Feature engineering
│   ├── model.py                  # CatBoost training
│   ├── predict.py                # Prediction pipeline
│   ├── odds.py / odds_monitor.py # Market odds fetching & movement tracking
│   ├── weather.py                # Weather features
│   └── config.py / utils.py
├── scripts/                # Pipelines, backtests, experiments, automation
│   ├── run_pipeline.py / run_collection.py
│   ├── train_ensemble.py / backtest_*.py / experiment_*.py
│   ├── export_dashboard_data.py
│   └── deploy.sh / entrypoint.sh / crontab
├── dashboard/              # Next.js + React dashboard
├── data/                   # raw / processed / master / model (.cbm)
├── run_predictions.py      # Main pipeline entry point
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+ (for the dashboard)
- An API key for The Odds API and Visual Crossing (optional: Anthropic, for generated match summaries)

### Installation

```bash
# Clone
git clone https://github.com/OlliePCK/AFL.git
cd AFL

# Python environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure secrets
cp .env.example .env            # then fill in your API keys
```

`.env` keys:

```
ODDS_API_KEY=...                # https://the-odds-api.com
VISUAL_CROSSING_API_KEY=...     # https://www.visualcrossing.com
ANTHROPIC_API_KEY=...           # optional, for match summaries
AFL_HOST_PORT=3000
```

### Usage

```bash
python run_predictions.py             # Update data, predict, refresh odds
python run_predictions.py --update    # Update dataset with latest results, then predict
python run_predictions.py --retrain   # Retrain on 2012–2024, validate on 2025
python run_predictions.py --optimize  # Optuna hyperparameter + feature optimisation
```

### Dashboard

```bash
cd dashboard
npm install
npm run dev          # http://localhost:3000
```

## Deployment

The project is containerised and runs on self-hosted infrastructure (Unraid). A GitHub Actions workflow builds the Docker image and pushes it to Docker Hub, and a cron schedule inside the container handles ongoing data collection, retraining and dashboard data export.

```bash
docker compose up -d --build
```

## License

Personal project — released under the MIT License.
