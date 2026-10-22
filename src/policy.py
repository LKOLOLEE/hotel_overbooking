"""
Analiza segmentów i rekomendacje polityk anulacji / depozytów.

Symulacja "what-if" dla wymogu przedpłaty w segmencie [SYNTETYCZNE założenia]:
  * odsetek anulacji spada o POLICY_CANCEL_REDUCTION (względnie)
  * liczba rezerwacji spada o POLICY_BOOKING_LOSS (część gości rezygnuje)
Wartość roczna = uratowane anulacje x wartość (późne: brak odsprzedaży,
wczesne: odsprzedaż z prawdopodobieństwem POLICY_EARLY_RESALE)
                - utracone rezerwacje x wartość x (1 - POLICY_REPLACEMENT)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src.eda import HOTEL_PL, LEAD_BINS, LEAD_LABELS
from src.plotting import COL, eur_axis, pct_axis, titles


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(lead_bucket=pd.cut(df["lead_time"], LEAD_BINS, labels=LEAD_LABELS).astype(str))
    d["late_cancel"] = ((d["is_cancel"] == 1) & (d["cancel_days_before"] <= 7)).astype(int)
    keys = ["hotel", "distribution_channel", "deposit_type", "lead_bucket"]
    g = d.groupby(keys).agg(
        rezerwacje=("booking_id", "size"),
        anulacje=("is_cancel", "mean"),
        anulacje_pozne=("late_cancel", "mean"),
        no_show=("is_no_show", "mean"),
        sr_przychod=("revenue", "mean"),
    ).reset_index()
    g["przychod_utracony"] = g["rezerwacje"] * (g["anulacje"] + g["no_show"]) * g["sr_przychod"]
    g["przychod_pozno_utracony"] = g["rezerwacje"] * (g["anulacje_pozne"] + g["no_show"]) * g["sr_przychod"]
    return g


def whatif_prepayment(seg: pd.DataFrame, years: float) -> pd.DataFrame:
    s = seg[(seg["deposit_type"] == "No Deposit") & (seg["rezerwacje"] >= C.POLICY_MIN_BOOKINGS)].copy()
    red, loss = C.POLICY_CANCEL_REDUCTION, C.POLICY_BOOKING_LOSS
    B, c, late = s["rezerwacje"], s["anulacje"], s["anulacje_pozne"]
    saved_late = B * (1 - loss) * late * red
    saved_early = B * (1 - loss) * (c - late) * red
    lost = B * loss * (1 - c)                         # zrezygnowali, a by przyjechali
    s["zmiana_pobytow"] = saved_late + saved_early - lost
    value = (saved_late + saved_early * (1 - C.POLICY_EARLY_RESALE)
             - lost * (1 - C.POLICY_REPLACEMENT)) * s["sr_przychod"]
    s["zmiana_przychodu_rocznie"] = value / years
    # mniejsza niepewność = mniej pustych pokoi z późnych anulacji
    s["mniej_poznych_anulacji_rocznie"] = s["rezerwacje"] * s["anulacje_pozne"] * red / years
    s["rekomendacja"] = np.where(
        s["zmiana_przychodu_rocznie"] > 0,
        "Wprowadzić przedpłatę / krótszy termin bezpłatnej anulacji",
        "Zachować obecne warunki (koszt utraty rezerwacji > korzyść)")
    out = s.sort_values("zmiana_przychodu_rocznie", ascending=False)
    out["hotel"] = out["hotel"].map(HOTEL_PL)
    return out.rename(columns={
        "hotel": "Hotel", "distribution_channel": "Kanał", "deposit_type": "Depozyt",
        "lead_bucket": "Wyprzedzenie (dni)", "rezerwacje": "Rezerwacje",
        "anulacje": "Odsetek anulacji", "anulacje_pozne": "Odsetek późnych anulacji",
        "no_show": "Odsetek no-show", "sr_przychod": "Śr. przychód z rezerwacji [EUR]",
        "przychod_utracony": "Przychód anulowany [EUR]",
        "przychod_pozno_utracony": "Przychód utracony późno [EUR]",
        "zmiana_pobytow": "Zmiana zrealizowanych pobytów",
        "zmiana_przychodu_rocznie": "Zmiana przychodu rocznie [EUR]",
        "mniej_poznych_anulacji_rocznie": "Mniej późnych anulacji rocznie",
        "rekomendacja": "Rekomendacja"})


def channel_table(df: pd.DataFrame, years: float) -> pd.DataFrame:
    g = df.groupby(["hotel", "distribution_channel"]).agg(
        rezerwacje=("booking_id", "size"),
        anulacje=("is_cancel", "mean"),
        no_show=("is_no_show", "mean"),
        przychod_anulowany=("revenue", lambda r: r[df.loc[r.index, "is_cancel"] == 1].sum()),
    ).reset_index()
    g = g[g["rezerwacje"] >= 100]
    g["przychod_anulowany"] /= years
    g["hotel"] = g["hotel"].map(HOTEL_PL)
    return g.rename(columns={"hotel": "Hotel", "distribution_channel": "Kanał",
                             "rezerwacje": "Rezerwacje", "anulacje": "Odsetek anulacji",
                             "no_show": "Odsetek no-show",
                             "przychod_anulowany": "Przychód anulowany rocznie [EUR]"})


def fig_channel_heat(df):
    d = df.assign(lb=pd.cut(df["lead_time"], LEAD_BINS, labels=LEAD_LABELS))
    t = d.groupby(["distribution_channel", "lb"], observed=True)["no_arrival"].agg(["mean", "size"])
    t = t[t["size"] >= 100]["mean"].unstack()
    t = t.loc[[c for c in ["Direct", "Corporate", "TA/TO", "GDS"] if c in t.index]]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", C.COLORS["seq"])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    im = ax.imshow(t.values, cmap=cmap, vmin=0, vmax=max(0.6, np.nanmax(t.values)), aspect="auto")
    ax.set_xticks(range(t.shape[1]), t.columns)
    ax.set_yticks(range(t.shape[0]), t.index)
    ax.grid(False)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.35 else COL["text"])
    ax.set_xlabel("Wyprzedzenie rezerwacji (dni)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    titles(ax, "Mapa ryzyka: kanał x wyprzedzenie",
           "Odsetek rezerwacji bez przyjazdu (pola z min. 100 rezerwacjami)")
    return fig


def fig_whatif(wi, top=10):
    s = wi.head(top).iloc[::-1]
    labels = [f"{h} | {k} | {w} dni" for h, k, w in zip(s["Hotel"], s["Kanał"], s["Wyprzedzenie (dni)"])]
    vals = s["Zmiana przychodu rocznie [EUR]"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(labels, vals, color=[COL["s1"] if v > 0 else COL["critical"] for v in vals], height=0.6)
    ax.axvline(0, color=COL["neutral"], lw=1)
    eur_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    titles(ax, "Gdzie przedpłata się opłaca (symulacja what-if)",
           f"Roczna zmiana przychodu netto; założenia: anulacje -{C.POLICY_CANCEL_REDUCTION:.0%}, "
           f"rezerwacje -{C.POLICY_BOOKING_LOSS:.0%}, odsprzedaż wczesnych anulacji {C.POLICY_EARLY_RESALE:.0%} [SYNTETYCZNE]")
    return fig
