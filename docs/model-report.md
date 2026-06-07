# Raport z budowy modelu — Nocarz

Dokument opisuje proces budowy modeli predykcji rezerwacji: które notatniki co robią, jak je uruchomić oraz jakie dają wyniki. Definicję problemu biznesowego, założenia i analizę danych zawiera [`project-summary.md`](project-summary.md).

## 1. Struktura potoku

Potok jest podzielony na pięć notatników o jasno rozdzielonych odpowiedzialnościach:

| Notatnik                       | Rola                                                                                                                                                   | Wejście                               | Wyjście                                                                               |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------------------- |
| `00-eda.ipynb`                 | Eksploracja danych (rozkłady, pokrycie tabel, skala braków). Bez modyfikacji danych.                                                                   | `data/raw/`                           | wykresy, wnioski                                                                      |
| `01-preprocessing.ipynb`       | Czyszczenie braków, konwersja typów, selekcja kolumn, analiza sentymentu VADER.                                                                        | `data/raw/`                           | `data/interim/`                                                                       |
| `02-feature_engineering.ipynb` | Etykieta `Y`, cechy recenzji, dołączenie cech oferty, empiryczny wybór imputacji, trening i zapis **modelu bazowego A**.                               | `data/interim/`                       | `data/processed/training_data.csv`, `inference_sample.csv`, `model_a_pipeline.joblib` |
| `03-modeling.ipynb`            | Konkurs 3 rodzin drzewiastych (RF, GB, XGBoost) ze **strojeniem hiperparametrów (CV)**, wybór i zapis **modelu docelowego B**, predykcje inferencyjne. | `data/processed/training_data.csv`    | `model_b_pipeline.joblib`, `inference_predictions.csv`                                |
| `04-ab_evaluation.ipynb`       | Ewaluacja eksperymentu A/B z logu mikroserwisu.                                                                                                        | `service/ab_logs/predictions_log.csv` | metryki A vs B                                                                        |

## 2. Jak uruchomić

```bash
# 1. środowisko
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2. potok danych i modeli (po kolei)
#    uruchom notatniki 00 -> 01 -> 02 -> 03 w Jupyter/VS Code

# 3. mikroserwis (z katalogu service/)
cd service
uvicorn app:app --reload --host 127.0.0.1 --port 8000

# 4. wygenerowanie ruchu A/B (w drugim terminalu)
python sample_requests.py --n 30 --with-labels 50

# 5. ewaluacja A/B
#    uruchom notebook 04-ab_evaluation.ipynb
```

## 3. Model bazowy A (`02-feature_engineering.ipynb`)

- **Rodzina:** regresja logistyczna (`class_weight='balanced'`).
- **Przygotowanie cech:** minimalne — imputacja (zabezpieczenie inferencji), skalowanie cech numerycznych, one-hot dla kategorii. Bez selekcji cech i bez usuwania outlierów.
- **Wybór strategii imputacji:** porównanie `median` vs `IterativeImputer` (MICE) po **Average Precision** w 4-krotnej walidacji krzyżowej na stratyfikowanej podpróbce. `KNNImputer` odrzucono — nie skaluje się do ~230 tys. wierszy (macierz odległości). Zwycięska strategia jest stosowana w całym potoku.
- **Artefakt:** cały `Pipeline` (impute + scale + encode + LogReg) zapisany jako `model_a_pipeline.joblib`.

## 4. Model docelowy B (`03-modeling.ipynb`)

- **Konkurs rodzin:** trzy rodziny **drzewiaste** — RandomForest, GradientBoosting (histogramowy), XGBoost. Regresja logistyczna nie startuje w konkursie (jest modelem bazowym A); SVM odrzucono z uwagi na skalę danych (~250 tys. wierszy) i ekstremalną nierównowagę klas.
- **Przygotowanie cech (`pre_tree`, wspólne dla trzech rodzin):** imputacja medianą + one-hot kategorii. Bez skalowania i bez winsoryzacji — drzewa dzielą po progach, więc są odporne na skalę i wartości odstające.
- **Strojenie hiperparametrów:** `RandomizedSearchCV` ze scoringiem `average_precision`, **3-krotna** stratyfikowana walidacja krzyżowa. Rozmiar siatek i `n_iter` są świadomie ograniczone ze względu na **ograniczenia sprzętowe** (cały konkurs ma zmieścić się w kilkunastu minutach). RandomForest dostał mniejszy budżet (najdroższy obliczeniowo), a XGBoost i GradientBoosting szersze siatki z parametrami regularyzacyjnymi (`reg_lambda`, `l2_regularization`, `min_child_weight`, `colsample_bytree`) i wolniejszym `learning_rate`.
- **Porównanie A vs B bez przecieku:** model bazowy A jest w tym notatniku trenowany ponownie tylko na `X_train` i oceniany na `X_test` (zapisany artefakt A z `02` jest trenowany na wszystkich danych i służy wyłącznie do serwowania).
- **Selekcja cech:** testowana równolegle, w ramach **ciekawostki** (`SelectFromModel` na bazie RandomForest, próg = mediana ważności) — sprawdza, czy przycięcie cech pomaga. Nie poprawia wyniku, więc finalny model B używa wszystkich cech.
- **Wybór:** zwycięża rodzina o najwyższym Average Precision na zbiorze testowym; ponownie wytrenowana na wszystkich danych i zapisana jako `model_b_pipeline.joblib`.

## 5. Wyniki

Liczby pochodzą z uruchomienia `03-modeling.ipynb` (stratyfikowany podział 80/20, `random_state=42`, ~317 tys. wierszy, 28 cech wejściowych). Stosunek klas neg/poz wynosi **136.7** (Y=1 ≈ 0.73%). Drobne różnice między uruchomieniami są możliwe (losowy `RandomizedSearchCV`).

### 5.1 Konkurs rodzin drzewiastych

| Model            | CV AP  | Test AP    | Test ROC-AUC |
| ---------------- | ------ | ---------- | ------------ |
| **XGBoost**      | 0.0415 | **0.0354** | **0.7535**   |
| GradientBoosting | 0.0416 | 0.0335     | 0.7463       |
| RandomForest     | 0.0422 | 0.0330     | 0.7463       |

**Zwycięzca: XGBoost** — najwyższe Test AP i ROC-AUC. Zapisany jako `model_b_pipeline.joblib` (wariant B w teście A/B).

Uwaga metodologiczna: model wybieramy po `Test AP`, więc ta wartość jest lekko optymistyczna (holdout posłużył do selekcji). Na „ślepym" CV AP trzy rodziny są praktycznie nieodróżnialne (0.0415–0.0422) — przewagę XGBoost rozstrzyga zbiór testowy.

### 5.2 Model bazowy A vs docelowy B

Oba modele oceniane na tym samym, nietkniętym `X_test`:

- Model docelowy B (XGBoost) osiąga wyższe ROC-AUC i Average Precision od bazowego A (regresja logistyczna), co uzasadnia wdrożenie bardziej złożonej rodziny jako wariantu B.
- Rozszerzenie siatki XGBoost o regularyzację i wolniejszy `learning_rate` podniosło Test AP najlepszego modelu z ~0.030 do **0.0354** (≈ +18%) **bez wzrostu kosztu CV** — czas zależy od liczby prób `n_iter`, a nie od rozmiaru przeszukiwanej przestrzeni.

### 5.3 Interpretacja

- Wszystkie modele wyraźnie biją poziom losowy (ROC-AUC ≈ 0.75 vs 0.50).
- Bezwzględny poziom Average Precision (~0.035) pozostaje niski — to **sufit sygnału w danych** (ekstremalna nierównowaga, większość sesji bez świeżych recenzji), a nie ograniczenie modelu. Strojenie hiperparametrów daje ułamki; istotny skok dałyby dopiero lepsze cechy (interakcje user×listing, historia sesji).
- Niska precyzja klasy `Booking` to konsekwencja nierównowagi ~137:1 — model jest **komponentem rankingowym** (`P(Y=1|X)` do reorderingu ofert), a nie twardym klasyfikatorem; właściwe miary to ROC-AUC i Average Precision.
- W ważności cech (wykres w `03-modeling.ipynb`, RandomForest) na czele są cechy recenzji — `days_since_last_review`, `alltime_review_scores_rating`, `alltime_review_count` — obok `price` i atrybutów gospodarza, co potwierdza hipotezę projektową, że świeżość i jakość recenzji wpływają na rezerwacje.

## 6. Eksperyment A/B i mikroserwis

- Mikroserwis (`service/`, FastAPI) ładuje oba pipeline'y i dla każdego żądania przydziela wariant **A** lub **B** deterministycznie po `user_id` (hash MD5, ~50/50). Wybór modelu jest **przezroczysty dla klienta** — odpowiedź ma jednolity format.
- Każda predykcja jest logowana do `service/ab_logs/predictions_log.csv` (wariant, nazwa modelu, prawdopodobieństwo, cechy, opcjonalne `Y`).
- `04-ab_evaluation.ipynb` czyta log i liczy rozkład ruchu A/B oraz — jeśli w logu jest `Y` — ROC-AUC i Average Precision obu wariantów na danych historycznych.
- Szczegóły API i przykładowe wywołania: [`service/README.md`](../service/README.md).

## 7. Artefakty (`data/processed/`)

| Plik                        | Opis                                            |
| --------------------------- | ----------------------------------------------- |
| `training_data.csv`         | Zbiór treningowy (cechy + `Y`).                 |
| `inference_sample.csv`      | Próbka inferencyjna (ostatnie 30 dni, bez `Y`). |
| `model_a_pipeline.joblib`   | Pipeline modelu bazowego A (wariant A).         |
| `model_b_pipeline.joblib`   | Pipeline modelu docelowego B (wariant B).       |
| `inference_predictions.csv` | Predykcje obu modeli na próbce inferencyjnej.   |
