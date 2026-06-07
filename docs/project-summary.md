# Optymalizacja silnika pozycjonującego oferty na portalu Nocarz

Celem zadania jest zbudowanie modelu wspomagającego proces rankingowania ofert podczas wyszukiwania na portalu Nocarz.

- Opis procesu budowy modeli, instrukcję uruchomienia i wyniki zawiera [`model-report.md`](model-report.md).
- Opis mikroserwisu i eksperymentu A/B: [`../service/README.md`](../service/README.md).

## 1. Definicja problemu biznesowego

Głównym problemem obecnego systemu jest zbyt niski wpływ nowo dodanych recenzji na pozycję oferty w wynikach wyszukiwania.

Zjawisko to wynika z matematycznej bezwładności średniej arytmetycznej. Jeśli ugruntowana oferta posiada już np. 100 opinii ze średnią 4.5, otrzymanie nowej, dobrej recenzji (5.0) podniesie jej ogólny wynik zaledwie do 4.505. Taka zmiana jest praktycznie niewidoczna dla algorytmu sortującego. Przez to miejsca, które w ostatnim czasie znacząco podniosły jakość i zbierają świetne recenzje, nie awansują w wynikach wyszukiwania, ponieważ ich nowe osiągnięcia są obniżne przez ciężar i ilość starych ocen.

## 2. Proponowane rozwiązanie

Zakładamy, że obecny silnik rankingowy przypisuje każdej pasującej do zapytania ofercie wynik ($BaseScore$) na podstawie dopasowania tekstowego oraz agregatów opinii (oceny cząstkowe i wolumeny czasowe). Obecny wzór może wyglądać tak:

$$BaseScore = w_1 \times \text{BM25}(query, offer) + w_2 \times QualityScore + w_3 \times VolumeScore$$

Nie znamy dokładnej implementacji, bo nie mamy dostępu do kodu wyszukiwarki ani do logów zapytań.

Aby nowe recenzje miały większy wpływ na pozycję oferty, nasz projekt ralizuje wdrożenie hybrydowego systemu pozycjonowania. Dodajemy nowy komponent ML, który estymuje prawdopodobieństwo rezerwacji $P(Y=1|X)$ na podstawie świeżości i jakości ostatnich recenzji. Stary algorytm nie jest zastępowany (zachowujemy weryfikację relewancji zapytania), lecz uzupełniany:

$$FinalScore = W_A \times BaseScore + W_B \times P(Y=1|X)$$

- $BaseScore$ - dotychczasowy wynik z istniejącego silnika (niemodyfikowany),
- $P(Y=1|X)$ - prawdopodobieństwo rezerwacji z naszego modelu,
- $W_A$, $W_B$ - wagi do ustalenia eksperymentalnie (test A/B).

Dzięki temu oferty z dobrymi _ostatnimi_ recenzjami awansują w rankingu, nawet jeśli ich ogólna średnia zmienia się minimalnie.

## 3. Kryteria sukcesu (cele biznesowe)

Głównym celem projektu jest poprawa efektywności procesu wyszukiwania. Mierzalne wskaźniki sukcesu to:

- **Wzrost konwersji:** odsetek sesji wyszukiwania zakończonych rezerwacją (Conversion Rate) wzrośnie o 25% względem wartości bazowej (np. z 60% do 75%).
- **Skrócenie czasu poszukiwań:** średnia liczba ofert przejrzanych przez użytkownika przed dokonaniem rezerwacji zmniejszy się o 25% (np. z 12 do 9 ofert).

## 4. Zadanie modelowania i inżynieria cech

Celem operacyjnym jest przewidzenie, czy użytkownik dokona rezerwacji danej oferty (klasyfikacja binarna). Model szacuje prawdopodobieństwo konwersji dla danej sesji: $P(Y=1 | X)$. Otrzymany wynik (w przedziale od 0 do 1) posłuży jako składnik $W_B \times P(Y=1|X)$ w funkcji rankingu.

**Model bazowy A** - regresja logistyczna (najprostszy sensowny model dla tego zadania).

**Model docelowy B** - najlepsza rodzina wyłoniona w konkursie trzech rodzin drzewiastych (RandomForest, GradientBoosting, XGBoost) z hiperparametrami dobranymi walidacją krzyżową. Konkurs wygrał **XGBoost**. Szczegóły procesu i wyniki zawarliśmy w [`model-report.md`](model-report.md).

### Definicja zmiennej celu (Y)

Każdy wiersz treningowy odpowiada zdarzeniu `view_listing` (użytkownik wyświetlił ofertę). Zmienna celu jest wyznaczana za pomocą `merge_asof` (forward, tolerancja 24h) między zdarzeniami `view_listing` a `book_listing` dla tej samej pary (użytkownik, oferta):

- $Y = 1$ (sukces): w ciągu 24 godzin od wyświetlenia oferty nastąpiła rezerwacja (`book_listing`) tej samej oferty przez tego samego użytkownika.
- $Y = 0$ (porażka): brak rezerwacji w oknie 24h.

### Inżynieria cech (X)

Dla każdego wiersza treningowego (sesja `view_listing` w momencie $T_0$) obliczany jest stan historyczny cech, z zachowaniem **reguły D+1**. Chodzi o to, że recenzja opublikowana w dniu $D$ staje się widoczna dopiero w dniu $D+1$, co zapobiega wyciekowi danych z przyszłości.

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

Z tabeli `Listings` dołączane są poprzez **inner join** (zachowywane są wyłącznie sesje dla ofert obecnych w tabeli). Wybrane zostały poprzez analizę AUC dla każdej cechy z osobna:

| Cecha                    | Typ                                                              |
| ------------------------ | ---------------------------------------------------------------- |
| `accommodates`           | numeryczna (liczba gości)                                        |
| `bathrooms`              | numeryczna (liczba łazienek)                                     |
| `bedrooms`               | numeryczna (liczba sypialni)                                     |
| `price`                  | numeryczna (cena za noc)                                         |
| `host_acceptance_rate`   | numeryczna (% rezerwacji akceptowanych przez gospodarza)         |
| `minimum_nights`         | numeryczna (minimalna długość pobytu wg polityki oferty)         |
| `minimum_maximum_nights` | numeryczna (górny limit polityki długości pobytu)                |
| `property_type`          | kategorialna (9 kategorii, rzadkie → "Other", braki → "Unknown") |
| `room_type`              | kategorialna (5 kategorii)                                       |
| `neighbourhood_cleansed` | kategorialna (12 dzielnic)                                       |
| `host_is_superhost`      | kategorialna (Yes / No / Unknown)                                |

To daje łącznie **28 cech wejściowych** (24 numeryczne + 4 kategorialne), które po kodowaniu one-hot rozszerzają się do ok. 53 kolumn. Braki w numerycznych cechach oferty (`price`, `bedrooms`, `bathrooms`, `host_acceptance_rate`) uzupełnia imputacja, której strategię (mediana vs `IterativeImputer`) wybieramy empirycznie po Average Precision w walidacji krzyżowej. Szczegóły także zawarliśmy w [`model-report.md`](model-report.md).

#### Świadome pominięcie cech z przecieku (data leakage)

Według naszej analizy (notatnik 00-eda), tabela `Listings` jest snapshotem z momentu pobrania danych, czyli z przyszłości względem analizowanych sesji. Dlatego kolumny zależne od późniejszych rezerwacji i recenzji (`review_scores_*`, `number_of_reviews*`, `reviews_per_month`, `availability_*`, `estimated_*`, `first/last_review` itd.) zostały odrzucone, mimo że są silnie skorelowane z `Y`. Wykorzystujemy wyłącznie atrybuty stabilne w czasie (typ, lokalizacja, pojemność, cena) oraz polityki/charakterystykę gospodarza (`host_is_superhost`, `host_acceptance_rate`, `minimum_nights`, `minimum_maximum_nights`), które w analizie EDA okazały się najsilniejsze. Natomiast, jak już zostało to wspomniane, sygnał recenzji budujemy samodzielnie, punktowo w czasie (VADER + reguła D+1).

## 5. Dane surowe

Projekt wykorzystuje zbiory danych z katalogu `data/raw/`:

| Tabela         | Wiersze | Kluczowe kolumny                                                                                                                 | Opis                                                                             |
| -------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `sessions.csv` | 470 445 | `user_id`, `timestamp`, `listing_id`, `action`                                                                                   | Logi sesji użytkowników (view_listing, book_listing, search i inne)              |
| `reviews.csv`  | 46 935  | `listing_id`, `date`, `comments`                                                                                                 | Recenzje tekstowe z datą publikacji (ziarnistość dzienna)                        |
| `listings.csv` | 3 646   | `id`, `property_type`, `room_type`, `accommodates`, `bathrooms_text`, `bedrooms`, `price`, `neighbourhood_cleansed` + ~70 innych | Snapshot ofert z atrybutami statycznymi                                          |
| `users.csv`    | -       | -                                                                                                                                | Dostępny, ale nieużywany w modelu (brak cech istotnych dla predykcji rezerwacji) |

## 6. Ocena wystarczalności danych

- Wszystkie kluczowe obserwacje ze wstępnej eksploracji są zawarte w notatniku 00-eda.
- Dane **pozwalają** zbudować model konwersji lepszy od losowego, ale ich siła predykcyjna jest ograniczona strukturalnie:
  - większość sesji dotyczy ofert bez żadnych widocznych recenzji (powód: wspomniany snapshot Listings),
  - tylko niewielki odsetek sesji ma recenzję w oknie 90-dniowym,
  - sentyment VADER bywa zaszumiony przez nieangielskie komentarze.
- Mimo ograniczeń cechy recenzji niosą realny sygnał (różnicują `Y=0` i `Y=1`). Nie jest on silny, ale używamy modelu jako dodatkowego komponentu rankingowego, a nie samodzielnego klasyfikatora.

## 7. Potok przetwarzania danych

Potok dzieli się na pięć notatników o rozłącznych odpowiedzialnościach. Pełna metodyka, parametry i wyniki znajdują się w [`model-report.md`](model-report.md):

| Notatnik                       | Rola                                                                                                                                                      | Wyjście                                                                |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `00-eda.ipynb`                 | Eksploracja: rozkłady, pokrycie tabel, skala braków, analiza przecieku i korelacji cech. Bez modyfikacji danych.                                          | wykresy, wnioski                                                       |
| `01-preprocessing.ipynb`       | Czyszczenie braków, konwersja typów, selekcja kolumn, grupowanie rzadkich kategorii, analiza sentymentu VADER.                                            | `data/interim/`                                                        |
| `02-feature_engineering.ipynb` | Etykieta `Y` (`merge_asof`, 24h), cechy recenzji (okno 90 dni + alltime), dołączenie cech oferty, wybór imputacji, trening i zapis **modelu bazowego A**. | `training_data.csv`, `inference_sample.csv`, `model_a_pipeline.joblib` |
| `03-modeling.ipynb`            | Konkurs rodzin drzewiastych ze strojeniem hiperparametrów (CV), wybór i zapis **modelu docelowego B**, predykcje inferencyjne.                            | `model_b_pipeline.joblib`, `inference_predictions.csv`                 |
| `04-ab_evaluation.ipynb`       | Ewaluacja eksperymentu A/B z logu mikroserwisu.                                                                                                           | metryki A vs B                                                         |
