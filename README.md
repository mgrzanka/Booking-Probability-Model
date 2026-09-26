# Booking Probability Model

**End-to-end ML project: from a business problem to a deployed A/B-tested microservice.**
Predicts `P(booking | session)` for a lodging marketplace, so that *recent* reviews can influence search ranking.

---

## The problem

On a rental portal, an offer's search position depends on the arithmetic mean of its reviews — and that mean is mathematically inert. An offer with 100 reviews averaging 4.5 that earns a new 5.0 review moves to 4.505. Places that genuinely improved recently never climb the results.

**Solution:** don't replace the ranker — add an ML term to it.

```
FinalScore = W_A · BaseScore + W_B · P(booking | X)
```

`BaseScore` is the existing engine (text match + review aggregates), untouched. The new component is a binary classifier estimating conversion probability from the **freshness and sentiment of recent reviews**. `W_A`, `W_B` are set by the A/B experiment.

## The ML task

| | |
|---|---|
| **Task** | Binary classification, extreme imbalance (**137:1**, 0.73% positives) |
| **Row** | One `view_listing` event at time `T₀` |
| **Label** | `Y=1` if the same user booked the same listing within 24h (`merge_asof`, forward, 24h tolerance) |
| **Features** | 28 inputs → ~53 after one-hot: review recency/volume, VADER aspect sentiment (7 aspects × 90-day window + all-time), static listing attributes |
| **Metric** | Average Precision (primary) + ROC-AUC — accuracy is meaningless at this imbalance |
| **Data** | 470k session events, 47k reviews, 3.6k listings → 317k training rows |

### Leakage was the hard part

`listings.csv` is a snapshot taken *after* the sessions occurred. Its `review_scores_*`, `number_of_reviews*`, `availability_*` and `estimated_*` columns correlate strongly with `Y` — because they already contain the outcome. **All of them are dropped.**

The review signal is instead rebuilt point-in-time from raw review text, under a **D+1 visibility rule**: a review published on day `D` only becomes a feature from day `D+1`. Only time-stable listing attributes (type, location, capacity, price) and host policies survive into the feature set.

## Results

Stratified 80/20 split, `random_state=42`, hyperparameters via `RandomizedSearchCV` (3-fold stratified CV, scoring = `average_precision`).

**Model competition → model B**

| Model | CV AP | Test AP | Test ROC-AUC |
|---|---|---|---|
| **XGBoost** ✅ | 0.0415 | **0.0354** | **0.7535** |
| GradientBoosting | 0.0416 | 0.0335 | 0.7463 |
| RandomForest | 0.0422 | 0.0330 | 0.7463 |

**Baseline A vs. target B** (same train split, same untouched test set)

| | ROC-AUC | Average Precision |
|---|---|---|
| A — LogisticRegression | 0.7240 | 0.0277 |
| **B — XGBoost** | **0.7535** | **0.0354** |

Reading these honestly: ROC-AUC ≈ 0.75 vs. 0.50 random is real, usable signal for a *ranking component*. The absolute AP of ~0.035 is a ceiling imposed by the data (most sessions concern listings with no visible reviews at all), not by the model — hyperparameter tuning barely moves it. Top features are exactly the ones the project hypothesised: `days_since_last_review`, `alltime_review_scores_rating`, `alltime_review_count`.

## Serving + A/B experiment

A FastAPI service loads both models as **self-contained pipelines** (imputation + encoding + scaling + classifier in one joblib artifact — no separate encoders to keep in sync) and splits traffic deterministically:

```
POST /predict → assign_variant(user_id) = MD5(user_id) % 2   # stable per user, ~50/50
             → variant A (LogisticRegression) | variant B (XGBoost)
             → log every prediction to ab_logs/predictions_log.csv
```

The variant's model name is read from the loaded pipeline, so swapping the winning model needs no service change. `04-ab_evaluation.ipynb` reads the log and scores both arms offline.

## Pipeline

| Notebook | Role | Output |
|---|---|---|
| `00-eda` | Distributions, coverage, missingness, **leakage & correlation analysis** | findings |
| `01-preprocessing` | Cleaning, type conversion, rare-category grouping, VADER sentiment | `data/interim/` |
| `02-feature_engineering` | Label via `merge_asof`, point-in-time review features, imputation choice, **baseline A** | `training_data.csv`, `model_a_pipeline.joblib` |
| `03-modeling` | 3-family competition + CV tuning, **target model B**, inference | `model_b_pipeline.joblib` |
| `04-ab_evaluation` | A/B log evaluation | A vs. B metrics |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. pipeline: run notebooks 00 → 01 → 02 → 03

# 2. service
cd service && uvicorn app:app --host 127.0.0.1 --port 8000

# 3. generate A/B traffic (second terminal)
python sample_requests.py --n 30 --with-labels 50

# 4. evaluate: notebook 04-ab_evaluation
```

**Stack:** Python · pandas · scikit-learn · XGBoost · vaderSentiment · FastAPI · Jupyter

**Full documentation (PL):** [`docs/project-summary.md`](docs/project-summary.md) — problem, success criteria, feature design · [`docs/model-report.md`](docs/model-report.md) — modelling process and results · [`service/README.md`](service/README.md) — API reference
