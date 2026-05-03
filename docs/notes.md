# Optymalizacja silnika pozycjonującego oferty na portalu Nocarz

Celem zadania jest zbudowanie modelu wspomagającego proces rankingowania ofert podczas wyszukiwaniu na portalu nocarz.

## Definicja problemu biznesowego

Głównym problemem obecnego systemu jest zbyt niski wpływ nowo dodanych recenzji na pozycję oferty w wynikach wyszukiwania.

Zjawisko to wynika z matematycznej bezwładności średniej arytmetycznej. Jeśli ugruntowana oferta posiada już np. 100 opinii ze średnią 4.5, otrzymanie nowej, entuzjastycznej recenzji (5.0) podniesie jej ogólny wynik zaledwie do 4.505. Taka zmiana jest praktycznie niewidoczna dla algorytmu sortującego. W efekcie, obiekty, które w ostatnim czasie znacząco podniosły jakość swoich usług i zbierają świetne recenzje, nie awansują w wynikach wyszukiwania, ponieważ ich nowe osiągnięcia są "zamrożone" przez ciężar i wolumen starych ocen.

## Proponowane rozwiązanie

Zakładamy, że obecnie system przypisuje wagę każdej z pasujących do zapytania ofercie na podstawie wzoru wykorzystującego agregaty opinii - oceny cząstkowe i wolumeny czasowe. Obecny algorytm może wyglądać mniej więcej tak:

$$BaseScore = (w_1 \times query-offerDescription BM25) + (w_2 \times QualityScore) + (w_3 \times VolumeScore)$$

Rozwiązaniem problemu jest wdrożenie hybrydowego systemu pozycjonowania. Do ostatecznego wyniku dodany zostanie nowy komponent optymalizowany z użyciem uczenia maszynowego, który estymuje prawdopodobieństwo rezerwacji na podstawie najnowszych recenzji użytkowniów.

Aby nie utracić weryfikacji dopasowania oferty do intencji użytkownika (relewancji zapytania), stary algorytm nie zostanie usunięty, lecz połączony z nowym modelem predykcyjnym:
$$FinalScore = (W_A \times BaseScore) + (W_B \times P(Y=1|X))$$
Wagi $W_A$ oraz $W_B$ zostaną zoptymalizowane eksperymentalnie (np. poprzez testy A/B w środowisku produkcyjnym), aby zmaksymalizować konwersję.

## Kryteria sukcesu (Cele biznesowe)

Wdrożenie nowego komponentu rankingowego ma na celu poprawę efektywności procesu wyszukiwania. Mierzalne wskaźniki sukcesu to:

- **Wzrost konwersji:** Odsetek sesji wyszukiwania zakończonych rezerwacją (Conversion Rate) wzrośnie o 25% względem wartości bazowej (np. z 60% do 75%).
- **Skrócenie czasu poszukiwań:** Średnia liczba ofert przejrzanych przez użytkownika przed dokonaniem rezerwacji zmniejszy się o 25% (np. z 12 do 9 ofert).

## Zadanie modelowania i inżynieria cech

Celem operacyjnym jest przewidzenie tego, czy użytkownik dokona rezerwacji danej oferty (klasyfikacja binarna). Zbudowany model (np. XGBoost, Random Forest lub regresja logistyczna) będzie szacował prawdopodobieństwo konwersji dla danej sesji: $P(Y=1 | X)$. Otrzymany wynik (w przedziale od 0 do 1) posłuży jako nowy składnik $W_B$ w głównej funkcji rankingu.

### Definicja zmiennej celu (Y)

Z tabeli `Sessions` wyodrębniona zostanie docelowa klasa decyzyjna:

- $Y = 1$ (Sukces): Zmienna `booking_id` nie jest pusta (użytkownik zarezerwował ofertę).
- $Y = 0$ (Porażka): Zmienna `booking_id` jest pusta.

### Inżynieria cech (X)

Dla każdego wiersza w tabeli `Sessions` obliczony zostanie stan historyczny wejściowych parametrów. Z tabeli `Reviews` wygenerowane zostaną m.in. następujące nowe zmienne:

- `days_since_last_review`: Czas (w dniach) od pojawienia się ostatniej opinii, pozwalający modelowi ocenić wpływ "świeżości".
- `reviews_in_last_7_days`: Agregacja pokazująca wolumen opinii z ostatniego tygodnia przed sesją.
- `recent_sentiment_score`: Średni sentyment opinii z ostatnich 7 dni przed analizowaną sesją (zastosowanie modelu klasyfikacji NLP na zawartości kolumny `comments`).

Wygenerowane zmienne tymczasowe połączone zostaną ze statycznymi danymi o ofercie z tabeli `Listings` (takimi jak cena, liczba sypialni, parametry gospodarza).

## 5. Analiza i walidacja dostarczonych danych (EDA)

Wstępna weryfikacja udostępnionych zbiorów danych wygenerowała następujące wnioski strukturalne i projektowe:

1. **Brak danych o reakcjach i zapytaniach:** Zgodnie z wytycznymi od interesariuszy, nie posiadamy logów zapytań wyszukiwarki (search history) ani reakcji użytkowników na poszczególne komentarze. To wymusza budowę modelu konwersji niezależnego od analizy słów kluczowych i opierającego się bezpośrednio na parametrach sesji.
2. **Rozdzielczość czasowa i Data Leakage:** Tabela `Reviews` przechowuje daty z ziarnistością dzienną (bez godziny), podczas gdy `Sessions` operuje na precyzyjnych znacznikach czasowych (`timestamp`). W celu zapobieżenia wyciekowi danych z przyszłości do modelu treningowego, łączenie danych zakłada widoczność nowej recenzji w systemie dopiero od dnia następnego po dodaniu ($D+1$).
3. **Braki danych w Listings:** Tabele ujawniają luki w ocenach cząstkowych ofert (np. brak `review_scores_value` dla nowych obiektów), które muszą być odpowiednio obsłużone w potoku imputacji danych przed podaniem ich do estymatora ML.
4. **Balans Zmiennej Celu:** Niezbędne jest zmierzenie stosunku sesji z wartością do sesji bez wartości w kolumnie `booking_id`, aby określić wariant równoważenia klas (np. Undersampling klasy większościowej).
