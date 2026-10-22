"""Raport dla Klienta - wersja komórkowa (Spyder / PyCharm / VS Code).
Uruchamiaj komórki po kolei (Spyder: Ctrl+Enter / Shift+Enter).
Wykresy pojawią się w panelu Plots, tabele w konsoli IPython."""

# %% [markdown]
# # Prognozowanie anulacji i inteligentny overbooking
# ### Raport dla Klienta (dane publiczne *Hotel Booking Demand* + elementy syntetyczne)
#
# Notatnik uruchamia **cały projekt** i pokazuje wyniki przeznaczone dla Klienta: diagnozę problemu,
# jakość modelu, backtest overbookingu, plan na najbliższe tygodnie, listę rezerwacji do kontaktu,
# analizę polityk anulacji, monitoring i business case.
#
# > Elementy oznaczone jako **SYNTETYCZNE** (koszt relokacji, popyt odrzucony w noce wyprzedane,
# > efekt kontaktu z gościem, efekt przedpłat) są założeniami - wszystkie znajdują się w `config.py`.
#
# Czas wykonania: ok. 1-2 minuty.

# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import pandas as pd
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

import config as C
from src.pipeline import run_all
from src.report import show, write_html, write_excel

R = run_all(as_of=C.DEMO_AS_OF)      # cały pipeline: dane -> modele -> backtest -> plan -> raport
T, F = R["tables"], R["figures"]

# %% [markdown]
# ## 1. Podsumowanie dla zarządu

# %%
show(T["kpi"])
show(F["business"])
show(T["business_items"])

# %% [markdown]
# ## 2. Diagnoza: skala problemu anulacji
# Ile rezerwacji nie kończy się pobytem, kiedy następują anulacje i jak często hotele są pełne.

# %%
show(T["eda_summary"], decimals={"Liczba rezerwacji": 0, "Śr. wyprzedzenie rezerwacji (dni)": 0}, transpose=True)
show(F["eda_trend"])
show(F["eda_lead"])
show(F["eda_segment"])
show(F["eda_timing"])
show(F["eda_occupancy"])

# %% [markdown]
# ## 3. Model ryzyka nieprzyjazdu
# Model ocenia ryzyko w konkretnym dniu przed przyjazdem (D-0 ... D-180) wyłącznie na podstawie
# informacji dostępnych w tym dniu (**migawki bez wycieku danych**). Walidacja czasowa:
# trening 07.2015-06.2016, kalibracja 07-12.2016, test 01-08.2017.

# %%
show(T["model_comparison"], decimals={"ROC_AUC": 3, "PR_AUC": 3, "Brier": 4, "LogLoss": 4})
show(F["model_roc"])
show(F["model_calibration"])
show(F["model_auc_horizon"])
show(T["model_by_horizon"], decimals={"ROC AUC (heurystyka)": 3, "ROC AUC (LightGBM)": 3,
                                      "Brier (heurystyka)": 4, "Brier (LightGBM)": 4})

# %% [markdown]
# ## 4. Dlaczego goście nie przyjeżdżają (SHAP)
# Wartości SHAP liczone natywnie przez LightGBM. Uwaga: wysokie ryzyko dla depozytu *Non Refund*
# to znany artefakt tego zbioru (takie rezerwacje są niemal zawsze anulowane).

# %%
show(F["shap_importance"])
show(F["shap_categories"])
show(F["shap_dependence"])

# %% [markdown]
# ## 5. Prognoza obłożenia
# Prognoza strukturalna (rezerwacje x prawdopodobieństwo przyjazdu + spodziewane nowe rezerwacje)
# jest korygowana modelem 2. stopnia; przedziały niepewności z reszt walidacji krzyżowej.

# %%
show(F["forecast_accuracy"])
show(T["forecast_accuracy"], decimals={"Liczba nocy": 0})

# %% [markdown]
# ## 6. Overbooking - backtest na 2017 r.
# Dla każdej nocy decyzja zapadała na D-3, wyłącznie na podstawie ówczesnej wiedzy.
# Porównanie: stan obecny, stały overbooking 5%, model ML, wyrocznia (górna granica).

# %%
show(F["ob_example"])
show(F["ob_cumulative"])
show(F["ob_bars"])
show(T["backtest_summary"])

# %% [markdown]
# ## 7. Plan overbookingu na kolejne 60 dni (tryb produkcyjny)

# %%
show(F["forward_plan"])
plan = T["forward_plan"][["hotel", "night", "horizon", "capacity", "in_house", "on_books", "forecast",
                          "occ_p10", "occ_p90", "p_full", "x_opt", "exp_gain", "rekomendacja"]]
show(plan, max_rows=20)

# %% [markdown]
# ## 8. Lista rezerwacji wysokiego ryzyka (przyjazd w ciągu 14 dni)

# %%
show(T["risk_summary"])
show(T["risk_list"], max_rows=25, decimals={"ID rezerwacji (pseudonim)": 0, "Dni do przyjazdu": 0, "Noce": 0})

# %% [markdown]
# ## 9. Działania prewencyjne - test A/B kontaktu z gościem  *(SYNTETYCZNY)*
# Efekt kontaktu nałożono syntetycznie na prawdziwe rezerwacje, aby pokazać metodę oceny:
# różnica odsetków z przedziałem ufności, efekt w decylach ryzyka i wybór grupy, którą opłaca się kontaktować.

# %%
show(T["outreach_overall"])
show(F["outreach_uplift"])
show(T["outreach_deciles"])

# %% [markdown]
# ## 10. Polityki anulacji i przedpłat

# %%
show(F["policy_heat"])
show(F["policy_whatif"])
show(T["policy_whatif"][["Hotel", "Kanał", "Wyprzedzenie (dni)", "Rezerwacje", "Odsetek anulacji",
                         "Odsetek późnych anulacji", "Zmiana przychodu rocznie [EUR]", "Rekomendacja"]], max_rows=12)
show(T["channels"])

# %% [markdown]
# ## 11. Monitoring modelu (dryf danych i stabilność)

# %%
show(F["drift"])
show(T["drift"], decimals={"PSI": 3}, max_rows=15)
show(F["monthly"])
show(T["monthly"], decimals={"ROC AUC": 3, "Brier": 4})

# %% [markdown]
# ## 12. Założenia i ograniczenia

# %%
show(T["assumptions"])

# %% [markdown]
# ## 13. Eksport: raport HTML i Excel
# Raport HTML (jeden plik z wykresami) można wysłać Klientowi; Excel zawiera pełne tabele.

# %%
html_path = write_html(R)
xlsx_path = write_excel(R)
print(html_path)
print(xlsx_path)
# import webbrowser; webbrowser.open(html_path.resolve().as_uri())   # otwórz w przeglądarce
