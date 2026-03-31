# March Madness 2026 Prediction Model

A machine learning pipeline for predicting NCAA Men's and Women's Basketball Tournament outcomes, built for the Kaggle March Machine Learning Mania competition.

---

## Overview

This project builds a bracket prediction model using historical game results, Elo ratings, tournament seedings, and season performance statistics. The pipeline progresses from a simple logistic regression baseline to an XGBoost model with engineered features.

**Final validation results (2024 tournament):**

| Model | Log Loss | Accuracy |
|---|---|---|
| Logistic Regression (WinPct + AvgPD) | 0.6847 | 59.7% |
| Logistic Regression + Nonlinear Features | 0.6370 | 58.2% |
| Logistic Regression + Seeds | 0.5778 | 70.1% |
| Logistic Regression + Seeds + Elo | 0.5794 | 70.1% |
| **XGBoost (all features)** | **0.6256** | **62.7%** |

---

## Data

All data is sourced from the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition.

- **381** men's teams, **379** women's teams
- **337,648** regular season games (1985–2026)
- **4,302** tournament games
- **519,144** matchups to predict in the submission file

---

## Pipeline

### 1. Elo Ratings

A custom Elo system is built from all regular season and tournament games (1985–2026):

- **Initial rating:** 1500
- **K-factor:** 20
- **Home court adjustment:** ±100 points
- **Season carry-over:** ratings regress 25% toward 1500 between seasons

**Top-rated teams (2026):**

| Men's | Women's |
|---|---|
| Houston (1821) | Connecticut (1913) |
| Duke (1819) | South Carolina (1893) |
| Arizona (1789) | UCLA (1869) |
| Connecticut (1785) | Texas (1858) |
| Gonzaga (1756) | LSU (1807) |

### 2. Feature Engineering

Features are constructed per matchup, always from Team1's perspective (Team1 ID < Team2 ID):

| Feature | Description |
|---|---|
| `WinPctDiff` | Difference in regular season win percentages |
| `AvgPDDiff` | Difference in average point differential |
| `WinPctDiff_sq` | Squared win percentage difference |
| `AvgPDDiff_sq` | Squared point differential difference |
| `Abs_WinPctDiff` | Absolute win percentage difference |
| `Abs_AvgPDDiff` | Absolute point differential difference |
| `Interaction` | `WinPctDiff × AvgPDDiff` |
| `SeedDiff` | Tournament seed difference (Team1 – Team2) |
| `EloDiff` | Elo rating difference at start of tournament |


---

## Setup

**Requirements:**
- Python 3.10+
- pandas, numpy, scikit-learn, xgboost

Download competition data from Kaggle and place CSVs in `data/raw/`.


---

## Ideas for Further Improvement

- **Richer stats:** Incorporate Ken Pomeroy efficiency ratings or Massey Ordinals for offensive/defensive splits
- **Recency weighting:** Downweight early season games when computing team statistics
- **Separate men's/women's models:** Dynamics differ across the two tournaments
- **Ensemble:** Blend logistic regression and XGBoost predictions
- **Hyperparameter tuning:** Optimize XGBoost via Optuna or Bayesian search
