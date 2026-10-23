# Instrukcja uruchomienia i utrzymania

## 1. Środowisko

```bash
python -m venv .venv
.venv\Scripts\activate                 # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt        # requirements-core.txt = tylko runtime (serwer / CI)
```

Jeśli terminal pokazuje `(base)` (Anaconda), przed aktywacją `.venv` wykonaj `conda deactivate`.
Żeby wyłączyć to na stałe: `conda config --set auto_activate_base false`.

## 2. PyCharm

1. **File → Open** i wskaż folder projektu (ten z `README.md`).
2. **Settings → Python Interpreter → Add Interpreter → Add Local Interpreter**, a następnie *Select existing* `.venv\Scripts\python.exe` albo *Generate new* (Virtualenv). Interpreter można też zmienić, klikając jego nazwę na pasku stanu w prawym dolnym rogu.
3. Gotowe konfiguracje uruchomieniowe z folderu `.run/` są na liście w prawym górnym rogu:
   - 1. Pipeline,
   - 2. Raport dzienny (replay),
   - 3. Raport dzienny (wybrana data),
   - 4. Testy,
   - 5. Przegląd modeli.

Przyciski Run korzystają z interpretera projektu, a nie z sesji terminala.

## 3. Raport dzienny i harmonogram

```bash
python daily_run.py --replay                                  # demo: dzisiejsza data → kolejny dzień okresu testowego 2017
python daily_run.py --as-of today --data <eksport_pms>.csv    # produkcja
python daily_run.py --replay --publish-dir "\\serwer\www\raport"
```

Dane publiczne kończą się w 08/2017, dlatego uruchomienia demo z harmonogramu używają `--replay`, a strona ma oznaczenie *DEMO*.
Brakujące modele trenują się automatycznie. Retrening raz w miesiącu: `run_pipeline.py`.

| Opcja | Konfiguracja | Uwagi |
|---|---|---|
| Harmonogram zadań Windows | `powershell -ExecutionPolicy Bypass -File scripts\zarejestruj_harmonogram.ps1 [-Time 07:30]` | Uruchamia raport o 8:00 i nadrabia, jeśli komputer był wyłączony. Log: `logs\daily_run.log`. Usunięcie: `scripts\usun_harmonogram.ps1` |
| GitHub Actions + Pages | `.github/workflows/raport-dzienny.yml`; Settings → Pages → Source: GitHub Actions | 06:00 UTC. **Strona publiczna – tylko demo** |
| Azure DevOps + Storage | `azure-pipelines-raport-dzienny.yml`; uzupełnij `STORAGE_ACCOUNT` i `AZURE_SERVICE_CONNECTION` | Dla danych Klienta: Static Web Apps z logowaniem Entra ID |
| Power BI | Zbiór danych oparty na `plan_overbookingu.xlsx` / `lista_ryzyka.xlsx` (lub tych samych tabelach w Snowflake) | Odświeżanie wg harmonogramu, dostęp zgodnie z uprawnieniami Klienta |

## 4. Przegląd modeli

`inspect_models.py`, uruchamiany komórka po komórce (`# %%`):

| Komórka | Zawartość |
|---|---|
| 1 | Pliki modeli i metryki na okresie testowym |
| 2–4 | Parametry LightGBM, zakodowane kategorie, ważność cech gain/split, pierwsze drzewo jako tekst |
| 5 | Kalibrator izotoniczny |
| 6 | Współczynniki regresji logistycznej i ilorazy szans |
| 7 | Tabela heurystyki segmentowej |
| 8 | Jedna rezerwacja we wszystkich modelach + czynniki SHAP |
| 9 | Scoring hipotetycznej rezerwacji (what-if) |

Pliki modeli w `outputs/models/`:
- `lgbm_model.txt` – wszystkie drzewa w postaci tekstowej,
- `calibrator.joblib` – kalibrator,
- `logistic_regression.joblib` – regresja logistyczna,
- `segment_heuristic.joblib` – heurystyka segmentowa,
- `occupancy_forecaster.joblib` – model obłożenia 2. stopnia,
- `settings.json` – ustawienia.

## 5. Dostosowanie do danych Klienta

1. Przygotuj eksport z PMS z tymi samymi kolumnami co `data/raw/hotel_bookings.csv` albo dopisz mapowanie w `src/data.py::clean`. Wymagane pola:
   - data rezerwacji, data przyjazdu, liczba nocy,
   - status i **data anulacji**,
   - kanał, segment, depozyt,
   - typ pokoju i cena.
2. W `config.py` ustaw `CAPACITY_OVERRIDE`, koszty i daty podziału czasowego.
3. Uruchom `run_pipeline.py` i zweryfikuj raport. Potem dodaj do harmonogramu `daily_run.py --as-of today --data ...`.
