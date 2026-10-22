"""Business case: roczny efekt finansowy, ROI i okres zwrotu."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src.overbooking import POLICY_NAMES
from src.plotting import COL, eur_axis, titles


def build(bt_summary: pd.DataFrame, bt_nights: int, outreach_annual: float,
          policy_annual: float) -> tuple[pd.DataFrame, dict]:
    factor = 365 / bt_nights
    ob_annual = bt_summary.loc[POLICY_NAMES["model"], "Wynik netto [EUR]"] * factor
    oracle_annual = bt_summary.loc[POLICY_NAMES["oracle"], "Wynik netto [EUR]"] * factor
    benefit = ob_annual + outreach_annual
    net_year = benefit - C.RUN_COST_EUR_PER_YEAR
    items = pd.DataFrame([
        {"Pozycja": "Overbooking oparty na modelu (backtest, annualizowany)", "EUR / rok": ob_annual,
         "Źródło": "dane publiczne + syntetyczny popyt odrzucony"},
        {"Pozycja": "Działania prewencyjne wobec ryzykownych rezerwacji", "EUR / rok": outreach_annual,
         "Źródło": "eksperyment SYNTETYCZNY"},
        {"Pozycja": "Koszt utrzymania rozwiązania", "EUR / rok": -C.RUN_COST_EUR_PER_YEAR,
         "Źródło": "założenie"},
        {"Pozycja": "RAZEM (bez zmian polityk)", "EUR / rok": net_year, "Źródło": ""},
        {"Pozycja": "Dodatkowo: potencjał zmian polityk przedpłat (do weryfikacji testem)",
         "EUR / rok": policy_annual, "Źródło": "symulacja what-if, SYNTETYCZNE założenia"},
        {"Pozycja": "Dla porównania: górna granica overbookingu (wyrocznia)",
         "EUR / rok": oracle_annual, "Źródło": "backtest"},
    ]).set_index("Pozycja")
    kpi = {
        "Korzyść netto rocznie [EUR]": net_year,
        "Koszt projektu [EUR]": C.PROJECT_COST_EUR,
        "ROI w 1. roku": (net_year - C.PROJECT_COST_EUR) / C.PROJECT_COST_EUR,
        "Okres zwrotu [miesiące]": C.PROJECT_COST_EUR / (net_year / 12) if net_year > 0 else float("nan"),
        "Korzyść na hotel rocznie [EUR]": net_year / 2,
    }
    return items, kpi


def fig_waterfall(items: pd.DataFrame):
    parts = items.iloc[:3]
    vals = parts["EUR / rok"].to_numpy()
    labels = ["Overbooking", "Działania\nprewencyjne", "Utrzymanie"]
    start = 0.0
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, v in enumerate(vals):
        ax.bar(i, v, bottom=start, color=COL["good"] if v >= 0 else COL["critical"], width=0.6)
        ax.text(i, start + v + (abs(vals).max() * 0.03 if v >= 0 else -abs(vals).max() * 0.08),
                f"{v / 1000:+,.0f} tys. €".replace(",", " "), ha="center", fontsize=9, color=COL["text"])
        start += v
    ax.bar(len(vals), start, color=COL["s1"], width=0.6)
    ax.text(len(vals), start + abs(vals).max() * 0.03, f"{start / 1000:,.0f} tys. €".replace(",", " "),
            ha="center", fontsize=9, fontweight="bold", color=COL["text"])
    ax.set_xticks(range(len(vals) + 1), labels + ["Razem"])
    peak = max(np.cumsum(vals).max(), start)
    ax.set_ylim(min(0, np.cumsum(vals).min()) * 1.1, peak * 1.2)
    ax.axhline(0, color=COL["neutral"], lw=1)
    eur_axis(ax)
    ax.grid(axis="x", visible=False)
    titles(ax, "Roczny efekt finansowy (2 hotele)", "Bez potencjału zmian polityk przedpłat")
    return fig
