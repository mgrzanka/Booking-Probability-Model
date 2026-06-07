# Raport z budowy modelu

Ten raport opisuje proces budowy modeli predykcji rezerwacji: które notatniki co robią, jak je uruchomić oraz jakie dają wyniki.

## 1. Struktura potoku

Potok jest podzielony na pięć notatników o różnych odpowiedzialnościach:

| Notatnik                       | Rola                                                                                                                                               | Wejście                               | Wyjście                                                                               |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------- |
| `00-eda.ipynb`                 | Eksploracja danych (badanie znaczenia atrybutów, rozkłady, pokrycie tabel, skala braków). Bez modyfikacji danych.                                  | `data/raw/`                           | wykresy, wnioski                                                                      |
| `01-preprocessing.ipynb`       | Czyszczenie braków, konwersja typów, selekcja kolumn, analiza sentymentu VADER.                                                                    | `data/raw/`                           | `data/interim/`                                                                       |
| `02-feature_engineering.ipynb` | Etykieta `Y`, cechy recenzji, dołączenie cech oferty, wybór imputacji dla brakujących danych, trening i zapis **modelu bazowego A**.               | `data/interim/`                       | `data/processed/training_data.csv`, `inference_sample.csv`, `model_a_pipeline.joblib` |
| `03-modeling.ipynb`            | Konkurs 3 rodzin drzewiastych (RF, GB, XGBoost) ze strojeniem hiperparametrów (CV), wybór i zapis **modelu docelowego B**, predykcje inferencyjne. | `data/processed/training_data.csv`    | `model_b_pipeline.joblib`, `inference_predictions.csv`                                |
| `04-ab_evaluation.ipynb`       | Ewaluacja eksperymentu A/B z logu mikroserwisu.                                                                                                    | `service/ab_logs/predictions_log.csv` | metryki A vs B                                                                        |

## 2. Jak uruchomić

```bash
# 1. środowisko
python -m venv .venv
source .venv\bin\activate
pip install -r requirements.txt

# 2. potok danych i modeli: notatniki 00 -> 01 -> 02 -> 03

# 3. mikroserwis
cd service
uvicorn app:app --reload --host 127.0.0.1 --port 8000

# 4. wygenerowanie ruchu A/B (w drugim terminalu)
python sample_requests.py --n 30 --with-labels 50

# 5. ewaluacja A/B - notatnik 04-ab_evaluation.ipynb
```

## 3. Model bazowy A (`02-feature_engineering.ipynb`)

- **Rodzina:** regresja logistyczna (`class_weight='balanced'`).
- **Przygotowanie cech:** selekcja cech, imputacja, skalowanie cech numerycznych, one-hot dla kategorii.
- **Strategia imputacji:** najlepsze wyniki dał `median` (eksperyment w notatniku 02).
- **Artefakt:** cały `Pipeline` (impute + scale + encode + LogReg) zapisany jako `model_a_pipeline.joblib`.

## 4. Model docelowy B (`03-modeling.ipynb`)

- **Konkurs trzech drzewiastych modeli:** sprawdziliśmy RandomForest, GradientBoosting i XGBoost. SVM odrzuciliśmy ze względu na rozmiar danych (~250 tys. wierszy) i bardzo silną nierównowagę klas.
- **Przygotowanie cech:** imputacja braków medianą, one-hot encoding zmiennych kategorycznych, brak skalowania i winsoryzacji (drzewa działają na progach, więc są odporne na różną skalę cech i wartości odstające).
- **Strojenie hiperparametrów:** `RandomizedSearchCV` z metryką `average_precision` i 3-krotną stratyfikowaną walidacją krzyżową. Siatki i `n_iter` dobraliśmy oszczędnie ze wzgldu na ograniczenie sprzętowe. Szczegóły w notatniku 03.
- **Porównanie A vs B:** Model bazowy A wytrenowaliśmy znowu tylko na `X_train` i oceniliśmy na `X_test`, bo artefakt A z notatnika 02 był już wytrenowany na pełnym zbiorze (pozostał modelem serwowanym w mikroserwisie).
- **Wybór finalny:** najlepszy wynik na zbiorze testowym dał XGBoost, dlatego to on został ponownie wytrenowany na wszystkich danych i zapisany jako `model_b_pipeline.joblib`

## 5. Wyniki

Liczby pochodzą z uruchomienia `03-modeling.ipynb` (podział równy względem klasy Y, proporcje 80/20, `random_state=42`, ~317 tys. wierszy, 28 cech wejściowych, stosunek neg/poz wynosi 0.73%).

### 5.1 Wybieranie najlepszego modelu

| Model            | CV AP  | Test AP    | Test ROC-AUC |
| ---------------- | ------ | ---------- | ------------ |
| **XGBoost**      | 0.0415 | **0.0354** | **0.7535**   |
| GradientBoosting | 0.0416 | 0.0335     | 0.7463       |
| RandomForest     | 0.0422 | 0.0330     | 0.7463       |

### 5.2 Model bazowy A vs docelowy B

Oba modele oceniliśmy na tym samym zbiorze testowym. Model docelowy B (XGBoost) osiągnął wyższe ROC-AUC i Average Precision od bazowego A (regresja logistyczna). Było to zgodne z tym, co przewidywaiśmy, bo jest to model bardziej złożony i lepiej modelujący jakikolwiek nieliniowe zależności.

### 5.3 Interpretacja

- Wszystkie modele wyraźnie biją poziom losowy (ROC-AUC ≈ 0.75 vs 0.50).
- Bezwzględny poziom Average Precision (~0.035) pozostaje niski ze względu na ekstremalną nierównowaga i fakt, że większość sesji jest bez świeżych recenzji, a nie przez ograniczenie modelu. Jest to sufit sygnału w danych. Strojenie hiperparametrów nie daje dużego zysku.
- Niska precyzja klasy `Booking` to konsekwencja nierównowagi ~137:1. Dlatego używamy miar ROC-AUC i Average Precision.
- W ważności cech (wykres w `03-modeling.ipynb`, RandomForest) na czele są cechy recenzji - `days_since_last_review`, `alltime_review_scores_rating`, `alltime_review_count`. Pozostałe - `price` i atrybutów gospodarza mają mniejsze znaczenie. Potwierdza to hipotezę projektową, że nowe recenzje mocno wpływają na rezerwacje.

## 6. Eksperyment A/B i mikroserwis

- Mikroserwis (`service/`, FastAPI) ładuje oba pipeline'y i dla każdego żądania przydziela wariant **A** lub **B** deterministycznie po `user_id` (hash MD5, ~50/50).
- Każda predykcja jest logowana do `service/ab_logs/predictions_log.csv`.
- `04-ab_evaluation.ipynb` czyta log i liczy rozkład ruchu A/B oraz, jeśli w logu jest `Y`, ROC-AUC i Average Precision obu wariantów na danych historycznych.
- Szczegóły API i przykładowe wywołania: [`service/README.md`](../service/README.md).

## 7. Artefakty (`data/processed/`)

| Plik                        | Opis                                            |
| --------------------------- | ----------------------------------------------- |
| `training_data.csv`         | Zbiór treningowy (cechy + `Y`).                 |
| `inference_sample.csv`      | Próbka inferencyjna (ostatnie 30 dni, bez `Y`). |
| `model_a_pipeline.joblib`   | Pipeline modelu bazowego A (wariant A).         |
| `model_b_pipeline.joblib`   | Pipeline modelu docelowego B (wariant B).       |
| `inference_predictions.csv` | Predykcje obu modeli na próbce inferencyjnej.   |
