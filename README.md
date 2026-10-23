# Prognozowanie anulacji i inteligentny overbooking

**Revenue management w hotelarstwie · Proof of concept Data Science**

Predykcja anulacji i no-show dla każdej rezerwacji. Wynik przekładany jest na rekomendację
overbookingu na każdą noc oraz codzienną listę rezerwacji do kontaktu, zanim zostaną anulowane.

---

## Problem biznesowy

| | Hotel miejski | Resort |
|---|---|---|
| Rezerwacje anulowane | 41% | 27% |
| W tym w ostatnich 7 dniach przed przyjazdem | 12% | 12% |
| Przychód anulowany późno (≤ 7 dni) | 1,17 mln € | 0,37 mln € |

Hotele zwykle kompensują anulacje jednym, stałym procentem overbookingu. W noce wysokiego popytu
pokoje zostają przez to puste, a w inne noce goście są relokowani. Nikt nie wie, **które**
rezerwacje są zagrożone, więc nikt nie podejmuje wobec nich działań.

## Podejście

1. **Model ryzyka.** LightGBM z kalibracją izotoniczną ocenia każdą aktywną rezerwację w dowolnym momencie od D-180 do D-0. Używa wyłącznie informacji dostępnych w danym dniu (migawki bez wycieku danych).
2. **Prognoza obłożenia.** Prawdopodobieństwa dla poszczególnych rezerwacji są agregowane do rozkładu końcowego obłożenia każdej nocy (rozkład Poissona-dwumianowy + pickup). Następnie koryguje je model 2. stopnia (krzywa rezerwacji) z przedziałami konformalnymi.
3. **Optymalizacja overbookingu.** Dla każdej nocy liczba dodatkowych rezerwacji wynika z bilansu wartości pokoju (ADR + wydatki dodatkowe) i kosztu relokacji gościa.
4. **Działania prewencyjne.** Przy każdej rezerwacji wysokiego ryzyka podawane są główne czynniki SHAP i zalecane działanie. Próg kontaktu wyznacza analiza upliftu z testu A/B.

## Kluczowe wyniki (test poza próbą, styczeń–sierpień 2017)

| Miara | Stan obecny | Rozwiązanie |
|---|---|---|
| ROC AUC (poziom rezerwacji) | 0,783 (heurystyka segmentowa) | **0,802** |
| Brier score | 0,0785 | **0,0754** |
| Wynik netto overbookingu, 2 hotele | −112 tys. € (stałe 5%) | **+68 tys. €** (37% górnej granicy) |
| Relokowani goście | 697 | **205** |
| Korzyść netto w skali roku¹ | – | **≈ 118 tys. €**, zwrot ≈ 3 miesiące |

¹ Obejmuje założenia syntetyczne (koszt relokacji, popyt odrzucony, efekt kontaktu z gościem), wszystkie opisane w `config.py`.
Przed podjęciem zobowiązań należy je zweryfikować na danych Klienta.

---

## Szybki start

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py                               # ok. 1-2 min, otwiera raport dla Klienta
python daily_run.py --replay --open                  # dzienny raport operacyjny (tryb demo)
```

Konfiguracja PyCharma, harmonogram i przegląd modeli: **[docs/uruchomienie.md](docs/uruchomienie.md)**.

## Rezultaty projektu

| Plik | Odbiorca |
|---|---|
| `outputs/reports/raport_klienta.html` | Zarząd: wnioski, model, backtest, business case, założenia |
| `outputs/reports/wyniki_projektu.xlsx` | Analitycy: wszystkie tabele, backtest noc po nocy |
| `outputs/site/index.html` (+ archiwum) | Revenue i recepcja: dzienny plan i lista kontaktów, gotowe do publikacji |
| `notebooks/raport_klienta.ipynb` | Przegląd całej analizy krok po kroku |
| `inspect_models.py` | Data science: drzewa, współczynniki, kalibracja, SHAP, what-if |

## Struktura repozytorium

```
config.py            parametry i wszystkie założenia [SYNTETYCZNE]
run_pipeline.py      pełny przebieg: trening, ocena, raporty
daily_run.py         dzienny scoring → plan, lista kontaktów, strona
inspect_models.py    przegląd modeli (komórka po komórce)
src/                 dane · cechy · modele · obłożenie · overbooking · prewencja · polityki · monitoring · raport
notebooks/           notatnik dla Klienta (+ wersja komórkowa dla Spyder/PyCharm)
scripts/ .github/ azure-pipelines-*.yml   harmonogram (Windows, GitHub Actions, Azure DevOps)
tests/               testy jednostkowe (pytest)
```

## Uwagi metodyczne

- **Walidacja wyłącznie w czasie.** Trening na przyjazdach do 06/2016, kalibracja na 07–12/2016, test na 01–08/2017.
- **Kontrola wycieku danych.** Wyłączono `country` (w tym zbiorze uzupełniany przy zameldowaniu) i wszystkie pola finalizowane po rezerwacji. Wyłączono też ADR i miesiąc przyjazdu, bo powodowały przeuczenie na sezon i poziom cen; z nimi AUC na teście spadał z 0,802 do 0,787.
- **Punkty odniesienia.** Modele porównano z heurystyką segmentową (stan obecny), regresją logistyczną i wyrocznią jako górną granicą. Backtest używa wspólnych liczb losowych.
- **Monitoring.** Śledzone są wskaźnik PSI (Population Stability Index) i kalibracja w kolejnych miesiącach. Dryf struktury agentów jest widoczny już w 2017 r. (PSI 0,75), dlatego zalecany jest comiesięczny retrening.

## Dane

[Hotel Booking Demand](https://doi.org/10.1016/j.dib.2018.11.126) (Antonio, Almeida i Nunes, 2019; CC BY 4.0):
119 390 rezerwacji z dwóch hoteli w Portugalii, przyjazdy 07/2015–08/2017. Elementy, których brakuje
w danych publicznych (liczba pokoi, koszt relokacji, popyt odrzucony, efekt kontaktu, reakcja na
przedpłaty), są syntetyczne i w pełni sparametryzowane w `config.py`.

## Ograniczenia i kolejne kroki

- **Zakres:** dwa hotele, poziom hotelu (a nie typu pokoju), wyłącznie dane historyczne. Kolejny krok to pilotaż na danych z PMS Klienta, z rzeczywistymi pojemnościami i kosztami.
- **Elementy syntetyczne:** należy je zastąpić zmierzonym kosztem relokacji, odmowami z silnika rezerwacyjnego i prawdziwym testem A/B kontaktu z gośćmi.
- **Annualizacja:** backtest obejmuje styczeń–sierpień, więc wynik roczny może być zawyżony. Trzeba go potwierdzić na pełnym sezonie.
- **Skalowanie:** model centralny z cechami hotelu dla wdrożenia w sieci oraz integracja wyników dziennych z Power BI lub Snowflake.
