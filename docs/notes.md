# Optymalizacja silnika pozycjonującego oferty na portalu Nocarz

Celem zadania jest zbudowanie modelu wspomagającego proces rankingowania ofert podczas wyszukiwaniu na portalu nocarz.

## 1. Definicja problemu biznesowego

Głównym problemem obecnego systemu jest zbyt niski wpływ nowo dodanych recenzji na pozycję oferty w wynikach wyszukiwania.

Zjawisko to wynika z matematycznej bezwładności średniej arytmetycznej. Jeśli ugruntowana oferta posiada już np. 100 opinii ze średnią 4.5, otrzymanie nowej, entuzjastycznej recenzji (5.0) podniesie jej ogólny wynik zaledwie do 4.505. Taka zmiana jest praktycznie niewidoczna dla algorytmu sortującego. W efekcie, obiekty, które w ostatnim czasie znacząco podniosły jakość swoich usług i zbierają świetne recenzje, nie awansują w wynikach wyszukiwania, ponieważ ich nowe osiągnięcia są "zamrożone" przez ciężar i wolumen starych ocen.

## 2. Proponowane rozwiązanie

Zakładamy, że obecnie system przypisuje wagę każdej z pasujących do zapytania ofercie na podstawie wzoru wykorzystującego agregaty opinii - oceny cząstkowe i wolumeny czasowe. Obecny algorytm może wyglądać mniej więcej tak:

$$BaseScore = (w_1 \times query-offerDescription BM25) + (w_2 \times QualityScore) + (w_3 \times VolumeScore)$$

Rozwiązaniem problemu jest wdrożenie hybrydowego systemu pozycjonowania. Do ostatecznego wyniku dodany zostanie nowy komponent optymalizowany z użyciem uczenia maszynowego, który estymuje prawdopodobieństwo rezerwacji na podstawie najnowszych recenzji użytkowniów.

Aby nie utracić weryfikacji dopasowania oferty do intencji użytkownika (relewancji zapytania), stary algorytm nie zostanie usunięty, lecz połączony z nowym modelem predykcyjnym:
$$FinalScore = (W_A \times BaseScore) + (W_B \times P(Y=1|X))$$
Wagi $W_A$ oraz $W_B$ zostaną zoptymalizowane eksperymentalnie (np. poprzez testy A/B w środowisku produkcyjnym), aby zmaksymalizować konwersję.

## 3. Kryteria sukcesu (Cele biznesowe)

Wdrożenie nowego komponentu rankingowego ma na celu poprawę efektywności procesu wyszukiwania. Mierzalne wskaźniki sukcesu to:

- **Wzrost konwersji:** Odsetek sesji wyszukiwania zakończonych rezerwacją (Conversion Rate) wzrośnie o 25% względem wartości bazowej (np. z 60% do 75%).
- **Skrócenie czasu poszukiwań:** Średnia liczba ofert przejrzanych przez użytkownika przed dokonaniem rezerwacji zmniejszy się o 25% (np. z 12 do 9 ofert).

## 4. Zadanie modelowania i inżynieria cech

Celem operacyjnym jest przewidzenie tego, czy użytkownik dokona rezerwacji danej oferty (klasyfikacja binarna). Wytrenowane modele (XGBoost, GradientBoosting, RandomForest, Regresja Logistyczna) szacują prawdopodobieństwo konwersji dla danej sesji: $P(Y=1 | X)$. Otrzymany wynik (w przedziale od 0 do 1) posłuży jako nowy składnik $W_B$ w głównej funkcji rankingu.

### Definicja zmiennej celu (Y)

Każdy wiersz treningowy odpowiada zdarzeniu `view_listing` (użytkownik wyświetlił ofertę). Zmienna celu jest wyznaczana za pomocą `merge_asof` (forward, tolerancja 24h) między zdarzeniami `view_listing` a `book_listing` dla tej samej pary (użytkownik, oferta):

- $Y = 1$ (Sukces): W ciągu 24 godzin od wyświetlenia oferty nastąpiła rezerwacja (`book_listing`) tej samej oferty przez tego samego użytkownika.
- $Y = 0$ (Porażka): Brak rezerwacji w oknie 24h.

### Inżynieria cech (X)

Dla każdego wiersza treningowego (sesja `view_listing` w momencie $T_0$) obliczany jest stan historyczny cech, z zachowaniem **reguły D+1** — recenzja opublikowana w dniu $D$ staje się widoczna dopiero w dniu $D+1$, co zapobiega wyciekowi danych z przyszłości.

#### Cechy oparte na recenzjach

Analiza sentymentu VADER (aspektowa + globalna) przetwarza treść komentarzy na 7 wyników numerycznych per recenzja:

| Cecha                     | Opis                                                                                                                                   |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `days_since_last_review`  | Liczba dni od $T_0$ do najnowszej widocznej recenzji. Wartość -1 gdy oferta nie ma żadnych recenzji.                                   |
| `reviews_in_last_90_days` | Liczba recenzji widocznych w oknie 90 dni przed $T_0$.                                                                                 |
| `recent_review_scores_*`  | Średnie wyniki sentymentu VADER z okna 90-dniowego (7 kolumn: cleanliness, location, communication, checkin, value, accuracy, rating). |
| `alltime_review_count`    | Łączna liczba recenzji widocznych w momencie $T_0$.                                                                                    |
| `alltime_review_scores_*` | Średnie wyniki sentymentu VADER ze wszystkich widocznych recenzji (7 kolumn).                                                          |

#### Cechy statyczne oferty

Z tabeli `Listings` dołączane są poprzez **inner join** (zachowywane są wyłącznie sesje dla ofert obecnych w tabeli):

| Cecha                    | Typ                                           |
| ------------------------ | --------------------------------------------- |
| `accommodates`           | numeryczna (liczba gości)                     |
| `bathrooms`              | numeryczna (liczba łazienek)                  |
| `bedrooms`               | numeryczna (liczba sypialni)                  |
| `price`                  | numeryczna (cena za noc)                      |
| `property_type`          | kategorialna (8 kategorii, rzadkie → "Other") |
| `room_type`              | kategorialna (5 kategorii)                    |
| `neighbourhood_cleansed` | kategorialna (12 dzielnic)                    |

Łącznie wygenerowanych zostaje 24 cechy wejściowe (21 numerycznych + 3 kategorialne), które po kodowaniu one-hot rozszerzają się do 46 cech.

## 5. Dane surowe

Projekt wykorzystuje trzy zbiory danych z katalogu `data/raw/`:

| Tabela         | Wiersze | Kluczowe kolumny                                                                                                                 | Opis                                                                             |
| -------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `sessions.csv` | 470 445 | `user_id`, `timestamp`, `listing_id`, `action`                                                                                   | Logi sesji użytkowników (view_listing, book_listing, search i inne)              |
| `reviews.csv`  | 46 935  | `listing_id`, `date`, `comments`                                                                                                 | Recenzje tekstowe z datą publikacji (ziarnistość dzienna)                        |
| `listings.csv` | 3 646   | `id`, `property_type`, `room_type`, `accommodates`, `bathrooms_text`, `bedrooms`, `price`, `neighbourhood_cleansed` + ~70 innych | Snapshot ofert z atrybutami statycznymi                                          |
| `users.csv`    | —       | —                                                                                                                                | Dostępny, ale nieużywany w modelu (brak cech istotnych dla predykcji rezerwacji) |

### Kluczowe obserwacje ze wstępnej eksploracji (EDA)

1. **Brak danych o zapytaniach:** Nie posiadamy logów zapytań wyszukiwarki (search history) ani reakcji użytkowników na poszczególne komentarze. Model konwersji opiera się bezpośrednio na parametrach sesji i recenzjach.
2. **Rozdzielczość czasowa i Data Leakage:** Tabela `Reviews` przechowuje daty z ziarnistością dzienną (bez godziny), podczas gdy `Sessions` operuje na precyzyjnych znacznikach czasowych. Zastosowano regułę D+1 — recenzja widoczna dopiero od dnia następnego po publikacji.
3. **Braki danych w Sessions:** ~248k wierszy (~53%) bez `timestamp`, `user_id` lub `action`. Dodatkowo ~168k zdarzeń `view_listing` bez `listing_id` oraz ~14k `book_listing` bez `listing_id`. Wszystkie usunięte (nie da się jednoznacznie przypisać oferty ani czasu).
4. **Braki danych w Listings:** Kolumny `bedrooms`, `bathrooms` i `price` posiadają wartości null. Nie usuwano — imputacja medianą w etapie inżynierii cech. Usunięto jedynie wiersze bez `listing_id` (952 wierszy).
5. **Listings jako snapshot:** Tabela Listings stanowi snapshot aktualnego stanu ofert — historyczne oferty (usunięte z portalu) są niedostępne. W efekcie 74% sesji z logów dotyczy ofert nieobecnych w tabeli.
6. **Nierównowaga klas:** Stosunek rezerwacji do przeglądań wynosi ~137:1 (Y=1: 0.73%). Zastosowano ważenie strat (weighted loss) zamiast undersamplingu, aby nie tracić danych treningowych.
7. **Rzadkie kategorie:** Kolumna `property_type` zawierała 38 unikalnych wartości, z których wiele miało < 5 ofert. Zgrupowano kategorie z < 30 ofertami do "Other" (finalnie 8 kategorii).

## 6. Potok przetwarzania danych (Pipeline)

### 6.1 Notebook `00-eda.ipynb` — Eksploracja i czyszczenie danych

1. **Wczytanie danych surowych** z `data/raw/` (reviews, sessions, listings).
2. **Konwersja typów:**
   - `price` (string `"$1,098.00"` → float), `bathrooms_text` (`"1 bath"` → float).
   - Daty na `datetime`, identyfikatory na `Int64`.
3. **Selekcja kolumn:** Z tabeli Listings zachowano 8 kolumn (`listing_id`, `property_type`, `room_type`, `accommodates`, `bathrooms`, `bedrooms`, `price`, `neighbourhood_cleansed`).
4. **Grupowanie rzadkich kategorii:** Typy nieruchomości z < 30 ofertami → "Other" (z 38 kategorii do 8).
5. **Usuwanie braków danych:**
   - Sessions: usunięcie wierszy bez `timestamp`, `user_id`, `listing_id` lub `action` (~248k wierszy).
   - Reviews: `listing_id`, `date` i `comments` wymagane — usunięcie wierszy z brakami.
   - Listings: usunięcie wierszy bez `listing_id`; braki w `bedrooms`, `bathrooms`, `price` pozostawione do imputacji medianą w dalszym etapie.
6. **Analiza sentymentu VADER:** Aspektowa klasyfikacja komentarzy — 7 wyników per recenzja (cleanliness, location, communication, checkin, value, accuracy, rating). Kolumna `comments` usuwana po ekstrakcji.
7. **Zapis do `data/interim/`:** `reviews_processed.csv` (9 kolumn), `sessions_processed.csv` (4 kolumny), `listings_processed.csv` (8 kolumn).

### 6.2 Notebook `01-feature_engineering.ipynb` — Generowanie cech

1. **Zmienna celu Y:** `merge_asof` (forward, tolerancja 24h) między widokami a rezerwacjami. $Y=1$ gdy rezerwacja nastąpiła w ciągu 24h od wyświetlenia.
2. **`days_since_last_review`:** `merge_asof` (backward) — ostatnia widoczna recenzja przed sesją. Brak recenzji → -1.
3. **Cechy okna 90-dniowego:** Liczba recenzji + średni sentyment z okna 90 dni przed sesją.
4. **Cechy historyczne (alltime):** Łączna liczba recenzji + średni sentyment ze wszystkich recenzji widocznych w momencie sesji.
5. **Inner join z Listings:** Zachowanie tylko sesji z ofertami obecnymi w tabeli (73.6% wierszy). Imputacja brakujących wartości numerycznych medianą, kategorii — "Unknown".
6. **Zapis do `data/processed/`:** `training_data.csv` (317 277 wierszy × 28 kolumn), `inference_sample.csv` (508 wierszy, ostatnie 30 dni).

### 6.3 Notebook `02-modeling.ipynb` — Trenowanie i ewaluacja modeli

1. **Preprocessing:** Usunięcie kolumn identyfikacyjnych (`user_id`, `timestamp`, `listing_id`). Podział stratyfikowany 80/20 (train/test).
2. **Kodowanie:** `OneHotEncoder` dopasowany **wyłącznie na zbiorze treningowym** (`handle_unknown='ignore'` — bezpieczna obsługa nieznanych kategorii przy inferencji). Rozszerzenie z 24 do 46 cech.
3. **Skalowanie:** `StandardScaler` dopasowany na zbiorze treningowym, stosowany wyłącznie dla Regresji Logistycznej (solwer gradientowy wymaga porównywalnych skal cech).
4. **Równoważenie klas (weighted loss):**
   - `RandomForest`, `LogisticRegression` — `class_weight='balanced'`
   - `GradientBoosting` — `sample_weight` z `compute_sample_weight('balanced')`
   - `XGBoost` — `scale_pos_weight` = stosunek klas negatywnych do pozytywnych (~136.7)
5. **Inferencja:** Enkoder i skaler zapisane jako artefakty — identyczny potok transformacji stosowany na nowych danych.

## 7. Wyniki modelowania

### 7.1 Statystyki zbioru treningowego

| Parametr                          | Wartość                                   |
| --------------------------------- | ----------------------------------------- |
| Liczba wierszy                    | 317 277                                   |
| Liczba cech (po kodowaniu)        | 46                                        |
| Klasa Y=0 (brak rezerwacji)       | 314 973 (99.27%)                          |
| Klasa Y=1 (rezerwacja)            | 2 304 (0.73%)                             |
| Stosunek nierównowagi             | ~137:1                                    |
| Pokrycie recenzjami (alltime)     | 27.0% sesji ma jakąkolwiek recenzję       |
| Pokrycie recenzjami (okno 90 dni) | 6.3% sesji ma recenzję z ostatnich 90 dni |

### 7.2 Porównanie modeli

| Model              | ROC-AUC    | F1 (Booking) | Precision (Booking) | Recall (Booking) |
| ------------------ | ---------- | ------------ | ------------------- | ---------------- |
| **RandomForest**   | **0.6976** | 0.043        | 0.02                | 0.38             |
| LogisticRegression | 0.6791     | 0.035        | 0.02                | 0.47             |
| XGBoost            | 0.6777     | 0.043        | 0.02                | 0.44             |
| GradientBoosting   | 0.6602     | 0.043        | 0.02                | 0.41             |

**Najlepszy model:** RandomForest (200 drzew, max_depth=10, class_weight='balanced').

### 7.3 Ważność cech (Feature Importance — RandomForest)

| Pozycja | Cecha                                 | Ważność |
| ------- | ------------------------------------- | ------- |
| 1       | `days_since_last_review`              | 16.1%   |
| 2       | `alltime_review_scores_rating`        | 10.1%   |
| 3       | `alltime_review_count`                | 9.9%    |
| 4       | `alltime_review_scores_location`      | 8.9%    |
| 5       | `price`                               | 5.7%    |
| 6       | `recent_review_scores_rating`         | 4.8%    |
| 7       | `alltime_review_scores_communication` | 4.4%    |
| 8       | `reviews_in_last_90_days`             | 4.0%    |
| 9       | `alltime_review_scores_cleanliness`   | 3.7%    |
| 10      | `alltime_review_scores_accuracy`      | 3.7%    |

Cechy oparte na recenzjach dominują w pierwszej dziesiątce ważności — **potwierdza to hipotezę projektową**, że świeżość i jakość recenzji mają istotny wpływ na prawdopodobieństwo rezerwacji.

### 7.4 Interpretacja wyników

- **ROC-AUC ~0.68–0.70** — wszystkie modele są wyraźnie lepsze od losowego (0.50), co potwierdza obecność sygnału w cechach. Wartość poniżej 0.80 wynika z ograniczeń danych, nie z błędów w potoku.
- **Niska precyzja klasy Booking (2%)** — efekt ekstremalnej nierównowagi klas. Model nie nadaje się do bezpośredniej klasyfikacji, ale jako **komponent rankingowy** (użycie $P(Y=1|X)$ do reorderingu wyników) ROC-AUC na poziomie 0.70 zapewnia użyteczne różnicowanie ofert.
- **Ważenie strat** (weighted loss) istotnie poprawiło recall modeli GradientBoosting (z 0.01 do 0.41) i XGBoost (0.44), czyniąc je konkurencyjnymi wobec RandomForest.
- **Główne ograniczenie danych:** ~73% sesji dotyczy ofert bez żadnych recenzji (tabela Listings jest snapshotem — historyczne oferty usunięte), a jedynie 6.3% sesji ma recenzjami w oknie 90-dniowym. To strukturalnie ogranicza siłę predykcyjną modelu.

### 7.5 Predykcje inferencyjne

Próbka inferencji (508 sesji z ostatnich 30 dni):

- Średnie $P(Y=1|X)$: 0.41, rozstęp: 0.14–0.89.
- Wystarczające zróżnicowanie prawdopodobieństw do efektywnego reorderingu w formule $FinalScore$.

## 8. Artefakty projektu

Wytrenowany model i artefakty preprocessingu zapisane w `data/processed/`:

| Plik                        | Opis                                              |
| --------------------------- | ------------------------------------------------- |
| `RandomForest_model.joblib` | Wytrenowany model RandomForest                    |
| `onehot_encoder.joblib`     | OneHotEncoder dopasowany na zbiorze treningowym   |
| `standard_scaler.joblib`    | StandardScaler dopasowany na zbiorze treningowym  |
| `feature_columns.joblib`    | Lista nazw cech (46 kolumn po kodowaniu)          |
| `column_config.joblib`      | Konfiguracja kolumn numerycznych i kategorycznych |
| `inference_predictions.csv` | Predykcje P(Y=1 \| X) dla próbki inferencyjnej    |
| `training_data.csv`         | Zbiór treningowy (317 277 × 28)                   |
| `inference_sample.csv`      | Próbka inferencji (508 × 27)                      |

### Struktura katalogów

```
data/
  raw/              # Dane surowe (reviews.csv, sessions.csv, listings.csv, users.csv)
  interim/          # Po EDA: oczyszczone + sentyment VADER
  processed/        # Po feature engineering: gotowe do modelowania + artefakty
notebooks/
  00-eda.ipynb              # Eksploracja i czyszczenie
  01-feature_engineering.ipynb  # Generowanie cech
  02-modeling.ipynb          # Trenowanie i ewaluacja
docs/
  notes.md          # Dokumentacja projektu
```
