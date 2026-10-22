# Prognozowanie anulacji i inteligentny overbooking

Kompletny projekt Data Science dla hotelu (lub sieci hoteli): model ryzyka anulacji / no-show,
prognoza faktycznego obłożenia, optymalizacja overbookingu, lista rezerwacji do kontaktu,
analiza polityk przedpłat, monitoring modelu i business case. Wyniki dla Klienta są generowane
w Pythonie: notatnik Jupyter, raport HTML (jeden plik z wykresami) i skoroszyt Excel.

Dane: publiczny zbiór **Hotel Booking Demand** (Antonio, Almeida, Nunes, *Data in Brief* 2019,
licencja CC BY 4.0) - 119 390 rezerwacji z hotelu miejskiego i resortu w Portugalii,
przyjazdy 07.2015-08.2017. Plik jest dołączony w `data/raw/`; jeśli go usuniesz, skrypt pobierze go ponownie.

---

## Szybki start

```bash
cd hotel_overbooking
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python run_pipeline.py          # ok. 1-2 min; na końcu otwiera raport HTML w przeglądarce
```

Wymagany Python 3.9+. Projekt przetestowano na Python 3.13, pandas 3.0, LightGBM 4.7.

| Środowisko | Jak uruchomić |
|---|---|
| **Jupyter Notebook** | `jupyter notebook notebooks/raport_klienta.ipynb` → *Run All* |
| **Spyder** | Otwórz `notebooks/raport_klienta_cells.py`, katalog roboczy = folder projektu, uruchamiaj komórki (Shift+Enter). Wykresy w panelu *Plots*. |
| **PyCharm** | Otwórz folder projektu - gotowe konfiguracje uruchomieniowe pojawią się w prawym górnym rogu (szczegóły niżej) |
| **Terminal** | `python run_pipeline.py --as-of 2017-07-01` |

Codzienny raport (plan overbookingu + lista rezerwacji do kontaktu + strona HTML):

```bash
python daily_run.py --as-of 2017-07-10 --open   # konkretny dzień w danych
python daily_run.py --replay --open             # tryb DEMO do harmonogramu (patrz niżej)
```
tworzy `outputs/daily/<data>/` (Excel, PNG, `raport_dzienny.html`) oraz stronę `outputs/site/`
(`index.html` = najnowszy raport, `archiwum/`, `pliki/`) gotową do publikacji online.
Jeśli modele nie istnieją, `daily_run.py` wytrenuje je automatycznie.

Testy: `python -m pytest -q`

---

## PyCharm

1. **File → Open** → wskaż folder `hotel_overbooking` (ten z `README.md`) → *Trust Project*.
2. **Interpreter**: PyCharm zwykle sam wykrywa `.venv`. Jeśli nie: *Settings → Project → Python Interpreter
   → Add Interpreter → Add Local Interpreter → Select existing* → `hotel_overbooking\.venv\Scripts\python.exe`.
3. Folder projektu jest katalogiem głównym, więc importy `config` i `src.*` działają bez dodatkowych ustawień.
4. W prawym górnym rogu (lista *Run/Debug Configurations*) są gotowe konfiguracje z folderu `.run/`:
   `1. Pipeline - trening i raport`, `2. Raport dzienny - demo (replay)`, `3. Raport dzienny - wybrana data`, `4. Testy`,
   `5. Przegląd modeli (LightGBM, LogReg)`.
   Gdyby się nie pojawiły: prawy klik na `run_pipeline.py` → *Run*.
5. Notatnik `notebooks/raport_klienta.ipynb` otwiera się w PyCharmie, jeśli Twoja wersja obsługuje Jupyter;
   w przeciwnym razie użyj przeglądarki (`jupyter notebook`) albo konfiguracji 1 (raport HTML).

---

## Automatyzacja: raport codziennie o 8:00 i publikacja online

### Ważne: tryb demo vs produkcja
Dane publiczne kończą się 31.08.2017, więc „dzisiejszej” daty w nich nie ma. Dlatego harmonogram
używa trybu **`--replay`**: każdy prawdziwy dzień jest mapowany na kolejny dzień okresu testowego
(15.01-02.07.2017, w pętli) - raport codziennie „przesuwa się” o jeden dzień, jak w prawdziwym hotelu.
Strona ma wtedy oznaczenie *DEMO*. W produkcji zamiast `--replay` używasz:
```bash
python daily_run.py --as-of today --data <ścieżka do dzisiejszego eksportu z PMS>.csv
```
(eksport w tym samym formacie kolumn co `hotel_bookings.csv`; modele retrenuj raz w miesiącu: `run_pipeline.py`).

### Opcja A - Harmonogram zadań Windows (lokalny komputer lub serwer)
```bat
powershell -ExecutionPolicy Bypass -File scripts\zarejestruj_harmonogram.ps1            :: codziennie 08:00
powershell -ExecutionPolicy Bypass -File scripts\zarejestruj_harmonogram.ps1 -Time 07:30
```
Zadanie uruchamia `scripts\daily_run.bat` (log: `logs\daily_run.log`). Jeśli komputer był wyłączony
o 8:00, raport wygeneruje się zaraz po jego włączeniu. Usunięcie: `scripts\usun_harmonogram.ps1`.
Publikacja: `daily_run.bat --publish-dir "\\serwer\www\raport"` kopiuje stronę do folderu serwera WWW
(IIS / nginx). Uwaga: OneDrive/SharePoint nie wyświetlają plików HTML jako stron - pobierają je.

### Opcja B - GitHub Actions + GitHub Pages (bez własnego komputera, demo)
1. Wypchnij projekt do repozytorium GitHub (`.gitignore` pomija `.venv` i `outputs`).
2. *Settings → Pages → Source: GitHub Actions*.
3. Workflow `.github/workflows/raport-dzienny.yml` uruchamia się codziennie o 06:00 UTC
   (08:00 latem, 07:00 zimą); ręcznie: *Actions → Run workflow*. Adres strony: `https://<login>.github.io/<repo>/`.

GitHub może opóźnić start zaplanowanego zadania o kilka-kilkanaście minut i wyłącza harmonogram w publicznym
repozytorium po 60 dniach bez aktywności. **Strona GitHub Pages jest publiczna** - tylko do demo na danych publicznych.

### Opcja C - Azure DevOps + Azure (produkcja z kontrolą dostępu)
`azure-pipelines-raport-dzienny.yml`: harmonogram 06:00 UTC, cache modeli, publikacja strony do kontenera `$web`
konta Azure Storage (static website). Uzupełnij `STORAGE_ACCOUNT` i `AZURE_SERVICE_CONNECTION`.
Dla danych Klienta wymagających logowania zamiast publicznego static website użyj **Azure Static Web Apps**
z uwierzytelnianiem Microsoft Entra ID (dostęp tylko dla wskazanych osób).

### Opcja D - Power BI
Pliki `plan_overbookingu.xlsx` i `lista_ryzyka.xlsx` (albo te same tabele zapisane do bazy / Snowflake)
mogą zasilać raport Power BI z zaplanowanym odświeżaniem - Klient ogląda go w Power BI Service z jego uprawnieniami.

---

## Przegląd zbudowanych modeli

`inspect_models.py` (komórki `# %%` - PyCharm *Run Cell* / Spyder Shift+Enter, albo `python inspect_models.py`):

| Komórka | Co pokazuje |
|---|---|
| 1 | Pliki modeli i porównanie metryk 4 modeli na 2017 r. |
| 2-4 | **LightGBM**: parametry, liczba drzew, zakodowane kategorie, ważność cech (gain/split), pierwsze drzewo jako tekst |
| 5 | **Kalibrator** (regresja izotoniczna): surowy wynik → prawdopodobieństwo |
| 6 | **Regresja logistyczna**: współczynniki i ilorazy szans (OR) |
| 7 | **Heurystyka segmentowa**: tabela odsetków nieprzyjazdów w segmentach |
| 8 | Ta sama rezerwacja w trzech modelach + wyjaśnienie SHAP |
| 9 | Hipotetyczna rezerwacja „co jeśli” - zmień pola i uruchom ponownie |

Wykresy i tabele: `outputs/inspect/`. Pliki modeli w `outputs/models/`:
`lgbm_model.txt` (tekstowy zapis wszystkich drzew - można otworzyć w PyCharmie), `calibrator.joblib`,
`logistic_regression.joblib` (pipeline scikit-learn: skalowanie + one-hot + LogisticRegression),
`segment_heuristic.joblib`, `occupancy_forecaster.joblib` (model obłożenia 2. stopnia).
W kodzie: `CancellationModel.load()` oraz `joblib.load(...)`.

---

## Wyniki (gdzie szukać)

| Plik | Zawartość |
|---|---|
| `outputs/reports/raport_klienta.html` | Raport dla Klienta: podsumowanie, diagnoza, model, backtest, plan, lista ryzyka, polityki, monitoring, business case, założenia |
| `outputs/reports/wyniki_projektu.xlsx` | Wszystkie tabele (plan na 60 dni, lista ryzyka, backtest noc po nocy, metryki…) |
| `outputs/figures/*.png` | Wykresy |
| `outputs/tables/*.csv` | Tabele w CSV |
| `outputs/models/` | Model LightGBM, kalibrator, model obłożenia, ustawienia |
| `outputs/site/index.html` | Najnowszy raport dzienny (+ `archiwum/`, `pliki/`) - strona do publikacji online |
| `outputs/daily/<data>/` | Raport dzienny: Excel, wykres, `raport_dzienny.html` |

---

## Struktura projektu

```
hotel_overbooking/
├── config.py                 # WSZYSTKIE parametry i założenia syntetyczne (oznaczone [SYNTETYCZNE])
├── run_pipeline.py           # pełny przebieg + raport HTML/Excel
├── inspect_models.py         # przegląd modeli: LightGBM, kalibracja, regresja logistyczna, heurystyka
├── daily_run.py              # codzienny raport: plan + lista ryzyka + strona HTML (outputs/site)
├── requirements-core.txt     # biblioteki potrzebne na serwerze / w harmonogramie
├── scripts/                  # Harmonogram zadań Windows (.bat, rejestracja/usunięcie .ps1)
├── .github/workflows/        # GitHub Actions: codzienny raport -> GitHub Pages
├── azure-pipelines-raport-dzienny.yml   # Azure DevOps: codzienny raport -> Azure Storage
├── .run/                     # gotowe konfiguracje uruchomieniowe PyCharm
├── data/raw/hotel_bookings.csv
├── notebooks/
│   ├── raport_klienta.ipynb        # raport dla Klienta (Jupyter)
│   └── raport_klienta_cells.py     # to samo w komórkach # %% (Spyder/PyCharm)
├── src/
│   ├── data.py               # wczytanie, czyszczenie, podział czasowy
│   ├── features.py           # migawki bez wycieku danych + cechy
│   ├── modeling.py           # heurystyka, regresja logistyczna, LightGBM + kalibracja
│   ├── explain.py            # ocena modelu, SHAP, powody ryzyka dla rezerwacji
│   ├── occupancy.py          # obłożenie nocne, pojemność, pickup
│   ├── overbooking.py        # prognoza obłożenia (2 stopnie), optymalizacja, backtest, plan
│   ├── overbooking_plots.py
│   ├── actions.py            # dzienna lista rezerwacji do kontaktu
│   ├── outreach.py           # test A/B działań prewencyjnych [SYNTETYCZNY] + uplift
│   ├── policy.py             # segmenty, mapa ryzyka, what-if przedpłat
│   ├── monitoring.py         # PSI, stabilność jakości w czasie
│   ├── business_case.py      # ROI, okres zwrotu
│   ├── eda.py, plotting.py   # diagnoza i styl wykresów
│   ├── report.py             # raport HTML, Excel, wyświetlanie w notatniku
│   ├── daily_report.py       # dzienny raport HTML, strona z archiwum, tryb replay
│   └── pipeline.py           # orkiestracja
└── tests/test_core.py
```

---

## Metodyka w skrócie

**1. Migawki bez wycieku danych.** Dla każdej rezerwacji tworzone są „zdjęcia” stanu w dniach
D-180, D-120, …, D-1, D-0 przed przyjazdem - tylko gdy rezerwacja była wtedy aktywna (już złożona,
jeszcze nieanulowana; data anulacji pochodzi z `reservation_status_date`). Cel: czy gość ostatecznie
nie przyjechał (anulacja po tej dacie lub no-show).

**2. Wyłączone cechy (wyciek / dryf).** `country` (w tym zbiorze uzupełniany przy zameldowaniu -
anulowane rezerwacje mają nienaturalnie często „PRT”), `booking_changes`, `assigned_room_type`,
`required_car_parking_spaces`, `total_of_special_requests` (wartości końcowe). Dodatkowo cena (ADR)
i miesiąc/tydzień przyjazdu: przy roku historii model zapamiętywał sezon i dryf cen - po ich usunięciu
ROC AUC na 2017 r. wzrósł z ok. 0,787 do ok. 0,802.

**3. Walidacja czasowa.** Trening: przyjazdy do 06.2016, walidacja i kalibracja: 07-12.2016,
test: 01-08.2017. Porównanie z heurystyką segmentową (to, co robi dziś revenue manager)
i regresją logistyczną. Metryki: ROC AUC, PR AUC, Brier, LogLoss, błąd kalibracji (ECE).

**4. Prognoza obłożenia (2 stopnie).**
- strukturalna: goście w hotelu + Σ(1 − p) dla rezerwacji na książce + pickup jako odsetek
  wolnej pojemności (metoda „constrained pickup”), rozkład Poisson-dwumianowy;
- korekta modelem LightGBM 2. stopnia uczonym na okresie walidacyjnym (tam model anulacji działa
  poza próbą - na okresie treningowym jego AUC wynosi ~0,93, czyli byłby przeuczony);
- niepewność: empiryczne reszty z walidacji krzyżowej blokami tygodni (podejście konformalne).

**5. Optymalizacja overbookingu.** Dla każdej nocy: ile *X* dodatkowych rezerwacji sprzedać, aby
zmaksymalizować `v·E[min(R+Y, Cap)] − w·E[max(R+Y−Cap, 0)]`, gdzie `R` - obłożenie przy obecnej
praktyce, `Y ~ Bin(X, q)`, `q` - odsetek przyjazdów rezerwacji last-minute, `v` - ADR + wydatki
dodatkowe, `w` - koszt relokacji gościa.

**6. Backtest 2017.** Decyzja na D-3 dla każdej nocy, porównanie: brak zmian, stały overbooking 5%,
model ML, wyrocznia (pełna wiedza = górna granica). Wspólne liczby losowe dla uczciwego porównania.

**7. Działania prewencyjne.** Lista rezerwacji wysokiego ryzyka z 3 głównymi powodami (SHAP) i
zalecanym działaniem; próg kontaktu wyznaczony z (syntetycznego) testu A/B i modelu upliftu.

---

## Dane syntetyczne - co i dlaczego

Zbiór publiczny nie zawiera wszystkiego, czego wymaga wdrożenie. Brakujące elementy są
**syntetyczne** i w całości sterowane z `config.py`:

| Brak w danych | Rozwiązanie |
|---|---|
| Liczba pokoi | Maksymalne zrealizowane obłożenie (dane wyraźnie dobijają do sufitu) - `CAPACITY_OVERRIDE` |
| Popyt odrzucony w noce wyprzedane | Poisson(10% pojemności) w noce z obłożeniem ≥ 95% |
| Koszt relokacji gościa | 2 × ADR + 150 € |
| Wydatki dodatkowe gości | 35 € (resort) / 20 € (hotel miejski) na pokój-noc |
| Efekt kontaktu z gościem | Syntetyczny test A/B na prawdziwych rezerwacjach (max −30% anulacji dla gości „niezdecydowanych”) |
| Efekt wymogu przedpłaty | −40% anulacji, −8% rezerwacji; odsprzedaż wczesnych anulacji 80% |
| Koszty projektu | 30 tys. € wdrożenie, 6 tys. € rocznie utrzymanie |

W prawdziwym wdrożeniu: liczba pokoi z PMS, koszt relokacji z finansów, popyt odrzucony
z systemu rezerwacyjnego (zapytania bez dostępności), efekt kontaktu z prawdziwego testu A/B.

---

## Jak dostosować do danych hotelu

1. Zastąp `data/raw/hotel_bookings.csv` eksportem z PMS z tymi samymi kolumnami
   (lub dopisz mapowanie w `src/data.py::clean`). Kluczowe: data rezerwacji, data przyjazdu,
   długość pobytu, status, **data anulacji**, kanał, segment, depozyt, typ pokoju, cena.
2. W `config.py` ustaw `CAPACITY_OVERRIDE`, koszty i daty podziału czasowego.
3. `python run_pipeline.py` → zweryfikuj raport → codziennie `python daily_run.py --as-of today --data <eksport>.csv`.

## Ograniczenia

- 2 hotele z Portugalii, lata 2015-2017 - wyniki należy potwierdzić na danych Klienta.
- Analiza na poziomie hotelu, nie typu pokoju (brak danych o pojemności typów i upgrade'ach).
- Wynik backtestu (styczeń-sierpień) annualizowany proporcjonalnie - może zawyżać efekt.
- Dane dryfują: struktura agentów/pośredników w 2017 r. wyraźnie różni się od treningowej (PSI > 0,25),
  a kalibracja odchyla się w części miesięcy - w produkcji zalecany comiesięczny retrening i rekalibracja.
