"""Wykresy dla modułu overbookingu."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src.eda import HOTEL_PL
from src.overbooking import POLICY_NAMES
from src.plotting import COL, eur_axis, titles

POLICY_COLORS = {"static": COL["s2"], "model": COL["s1"], "oracle": COL["s3"],
                 "none": COL["neutral"]}


def fig_example_night(dec, forecaster):
    """Jak powstaje decyzja dla jednej nocy (rozkład obłożenia vs pojemność)."""
    cand = dec[(dec["x_opt"] > 0) & (dec["sold_out_night"])]
    r = (cand if len(cand) else dec).sort_values("exp_gain").iloc[-1]
    cap = int(r["capacity"])
    pmf_r = forecaster.pmf(r["forecast"], r["hotel"], int(r["horizon"]), cap)
    x = int(r["x_opt"])
    q = r["q_extra_show"]
    binom = np.array([1.0])
    for _ in range(x):
        b = np.zeros(len(binom) + 1)
        b[:-1] += binom * (1 - q)
        b[1:] += binom * q
        binom = b
    pmf_t = np.convolve(pmf_r, binom)
    k_r, k_t = np.arange(len(pmf_r)), np.arange(len(pmf_t))
    lo = max(int(r["forecast"]) - 25, 0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(k_r, pmf_r, where="mid", color=COL["neutral"], lw=2,
            label="Obłożenie bez dodatkowej sprzedaży")
    ax.step(k_t, pmf_t, where="mid", color=COL["s1"], lw=2,
            label=f"Po sprzedaży +{x} rezerwacji (rekomendacja)")
    ax.axvline(cap, color=COL["critical"], lw=1.2, ls="--")
    ax.text(cap + 0.5, ax.get_ylim()[1] * 0.9, f"pojemność {cap}", color=COL["text2"], fontsize=9)
    ax.set_xlim(lo, cap + 15)
    ax.set_xlabel("Liczba gości (zajętych pokoi + relokacje)")
    ax.set_ylabel("Prawdopodobieństwo")
    ax.legend(loc="upper left")
    p_walk = pmf_t[cap + 1:].sum()
    titles(ax, f"Przykładowa decyzja: {HOTEL_PL[r['hotel']]}, noc {r['night']:%d.%m.%Y} (decyzja na D-{int(r['horizon'])})",
           f"Oczekiwany zysk z +{x} rezerwacji: {r['exp_gain']:,.0f} EUR; ryzyko relokacji: {p_walk:.0%}".replace(",", " "))
    return fig


def fig_backtest_cumulative(dec):
    d = dec.sort_values("night").groupby("night")[[f"{k}_gain" for k in POLICY_NAMES]].sum().cumsum()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for key in ["oracle", "model", "static"]:
        ax.plot(d.index, d[f"{key}_gain"], color=POLICY_COLORS[key], label=POLICY_NAMES[key])
        ax.annotate(f"{d[f'{key}_gain'].iloc[-1] / 1000:,.0f} tys. €".replace(",", " "),
                    (d.index[-1], d[f"{key}_gain"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=COL["text"])
    ax.axhline(0, color=COL["neutral"], lw=1)
    eur_axis(ax)
    ax.set_ylabel("Skumulowany wynik netto vs stan obecny")
    ax.legend(loc="upper left")
    titles(ax, "Backtest 2017: model ML zarabia, stały overbooking traci",
           "Przychód z odzyskanych pokoi minus koszt relokacji gości (oba hotele)")
    return fig


def fig_backtest_bars(summary):
    s = summary.drop(index=POLICY_NAMES["none"])
    keys = [k for k in POLICY_NAMES if POLICY_NAMES[k] in s.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.barh(s.index, s["Wynik netto [EUR]"], color=[POLICY_COLORS[k] for k in keys], height=0.6)
    ax.axvline(0, color=COL["neutral"], lw=1)
    eur_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Wynik netto [EUR]", fontsize=10)
    ax = axes[1]
    ax.barh(s.index, s["Relokacje gości"], color=[POLICY_COLORS[k] for k in keys], height=0.6)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_title("Liczba relokowanych gości", fontsize=10)
    from matplotlib.ticker import MaxNLocator
    axes[0].xaxis.set_major_locator(MaxNLocator(5))
    for a, col in zip(axes, ["Wynik netto [EUR]", "Relokacje gości"]):
        span = s[col].abs().max()
        for i, v in enumerate(s[col]):
            a.text(v + (0.02 * span if v >= 0 else -0.02 * span), i, f"{v:,.0f}".replace(",", " "),
                   va="center", ha="left" if v >= 0 else "right", fontsize=9, color=COL["text"])
        lo, hi = min(s[col].min(), 0), max(s[col].max(), 0)
        a.set_xlim(lo - 0.45 * span * (lo < 0), hi + 0.2 * span)
    fig.tight_layout()
    return fig


def fig_forecast_accuracy(acc):
    hotels = acc.index.get_level_values(0).unique()
    fig, axes = plt.subplots(1, len(hotels), figsize=(11, 3.8), sharey=True)
    series = [("MAE: naiwna (rok temu)", COL["neutral"]), ("MAE: strukturalna", COL["s2"]),
              ("MAE: model 2. stopnia", COL["s1"])]
    for ax, hotel in zip(np.atleast_1d(axes), hotels):
        a = acc.loc[hotel]
        x = np.arange(len(a))
        for col, color in series:
            ax.plot(x, a[col], marker="o", ms=5, color=color, label=col.replace("MAE: ", ""))
        ax.set_xticks(x, a.index.astype(str))
        ax.set_title(HOTEL_PL[hotel], fontsize=10)
        ax.set_xlabel("Horyzont prognozy")
    np.atleast_1d(axes)[0].set_ylabel("Średni błąd (liczba pokoi)")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.suptitle("Trafność prognozy obłożenia - im bliżej przyjazdu, tym dokładniej",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", color=COL["text"])
    fig.tight_layout()
    return fig


def fig_forward_plan(plan, as_of):
    hotels = sorted(plan["hotel"].unique())
    fig, axes = plt.subplots(len(hotels), 1, figsize=(10, 6), sharex=True)
    for ax, hotel in zip(np.atleast_1d(axes), hotels):
        p = plan[plan["hotel"] == hotel]
        cap = int(p["capacity"].iloc[0])
        ax.fill_between(p["night"], p["occ_p10"], p["occ_p90"], color=C.COLORS["seq"][1],
                        alpha=0.7, lw=0, label="Przedział P10-P90")
        ax.plot(p["night"], p["forecast"], color=COL["s1"], label="Prognoza obłożenia")
        ax.plot(p["night"], p["in_house"] + p["on_books"], color=COL["neutral"], lw=1.2,
                ls=":", label="Rezerwacje na książce (z anulacjami)")
        ax.axhline(cap, color=COL["critical"], lw=1, ls="--")
        ob = p[p["rekomendacja"].str.startswith("Dopuść")]
        ax.scatter(ob["night"], np.full(len(ob), cap * 0.62), s=12, color=COL["good"],
                   label="Dopuszczalny overbooking", zorder=3)
        ax.set_ylim(cap * 0.6, cap * 1.25)
        ax.set_ylabel("Pokoje")
        ax.set_title(f"{HOTEL_PL[hotel]} (pojemność {cap})", fontsize=10, pad=4)
    np.atleast_1d(axes)[0].legend(loc="upper right", ncol=2, fontsize=8)
    fig.suptitle(f"Plan na {len(plan) // len(hotels)} dni od {pd.Timestamp(as_of):%d.%m.%Y}",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", color=COL["text"])
    fig.tight_layout()
    return fig
