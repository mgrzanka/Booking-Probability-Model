# Mikroserwis predykcji rezerwacji (A/B)

Lokalny serwis FastAPI do predykcji prawdopodobieństwa rezerwacji. Każdy request trafia losowo (po `user_id`) do jednego z dwóch modeli:

- **wariant A** -- model bazowy (regresja logistyczna),
- **wariant B** -- model docelowy (najlepsza rodzina wyłoniona w konkursie modeli).

Nazwa modelu w odpowiedzi i logu jest odczytywana z zapisanego pipeline'u, więc zmiana zwycięzcy modelu B nie wymaga zmian w serwisie.

## Wymagania

Modele muszą być wcześniej zapisane jako samowystarczalne pipeline'y do `data/processed/`:

- `model_a_pipeline.joblib` -- zapisywany przez `notebooks/02-feature_engineering.ipynb` (model bazowy A),
- `model_b_pipeline.joblib` -- zapisywany przez `notebooks/03-modeling.ipynb` (model docelowy B).

Każdy pipeline zawiera kompletny preprocessing (imputacja + kodowanie + skalowanie) i klasyfikator, więc serwis nie potrzebuje osobnych enkoderów ani skalerów.

## Uruchomienie

Z katalogu `service` (po aktywacji `venv`):

```bash
uvicorn app:app --reload --host 127.0.0.1 --port <numer portu>
```

W przykładach numer portu ustawiony jest na 8000.

Sprawdzenie zdrowia serwisu:

```bash
curl http://127.0.0.1:8000/health
```

## Przykładowy request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 40137464,
    "listing_id": 955594135974379648,
    "days_since_last_review": -1,
    "alltime_review_count": 0,
    "reviews_in_last_90_days": 0,
    "price": 1200.0,
    "property_type": "Unknown",
    "room_type": "Unknown",
    "accommodates": 2.0,
    "bathrooms": 1.0,
    "bedrooms": 1.0,
    "neighbourhood_cleansed": "Vesterbro-Kongens Enghave",
    "host_is_superhost": "No",
    "host_acceptance_rate": 90.0,
    "minimum_nights": 3.0,
    "minimum_maximum_nights": 365.0
  }'
```

Przykładowa odpowiedź:

```json
{
  "request_id": "f3b2c1a0-1234-5678-90ab-cdef12345678",
  "model_variant": "A",
  "model_name": "LogisticRegression",
  "booking_probability": 0.0123,
  "user_id": 40137464,
  "listing_id": 955594135974379648
}
```

## Log A/B

Każda predykcja jest zapisywana do:

`service/ab_logs/predictions_log.csv`

Kolumny logu: `timestamp`, `request_id`, `user_id`, `listing_id`, `model_variant`, `model_name`, `booking_probability`, wybrane cechy wejściowe oraz opcjonalnie `Y` (jeśli podano w requeście testowym).

## Generowanie przykładowych requestów

```bash
python sample_requests.py --n 30 --with-labels 50
```

Skrypt wysyła pierwsze N wierszy z `data/processed/inference_sample.csv` oraz dodatkowe requesty z `training_data.csv` (z polem `Y` do ewaluacji offline).

## Ewaluacja logu A/B

Po wygenerowaniu logu należy uruchomić notebook:

`notebooks/04-ab_evaluation.ipynb`
