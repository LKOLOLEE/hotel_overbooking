"""
Budowa zbioru "migawek" (snapshots) bez wycieku danych oraz cech modelu.

Idea: model ma przewidywać ryzyko w konkretnym momencie przed przyjazdem
(D-h). Dla każdej rezerwacji i każdego horyzontu h tworzymy wiersz tylko
wtedy, gdy rezerwacja była AKTYWNA w dniu D-h:
  * została już złożona:          lead_time >= h
  * nie była jeszcze anulowana:   anulacja nastąpiła w dniu D-h lub później
Cel: czy gość ostatecznie nie przyjechał (anulacja po D-h lub no-show).
Używamy wyłącznie informacji znanych w dniu D-h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from src.data import CATEGORICAL

# Uwaga: miesiąc/tydzień przyjazdu oraz cena (ADR) zostały świadomie
# WYŁĄCZONE z modelu. Przy ~1 roku historii treningowej model "zapamiętywał"
# sezon konkretnego roku, a ceny dryfują (inflacja, zmiany cenników) -
# w backteście obniżało to ROC AUC na okresie testowym (patrz README).
NUMERIC_FEATURES = [
    "lead_time", "days_to_arrival", "days_since_booking", "arrival_dow",
    "stays_in_weekend_nights", "stays_in_week_nights", "nights",
    "adults", "children", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "has_company", "days_in_waiting_list",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL

FEATURE_LABELS_PL = {
    "lead_time": "Wyprzedzenie rezerwacji (dni)",
    "days_to_arrival": "Dni do przyjazdu",
    "days_since_booking": "Dni od złożenia rezerwacji",
    "arrival_month": "Miesiąc przyjazdu",
    "arrival_week": "Tydzień przyjazdu",
    "arrival_dow": "Dzień tygodnia przyjazdu",
    "stays_in_weekend_nights": "Noce weekendowe",
    "stays_in_week_nights": "Noce w tygodniu",
    "nights": "Długość pobytu",
    "adults": "Liczba dorosłych",
    "children": "Liczba dzieci",
    "babies": "Liczba niemowląt",
    "is_repeated_guest": "Gość powracający",
    "previous_cancellations": "Wcześniejsze anulacje gościa",
    "previous_bookings_not_canceled": "Wcześniejsze zrealizowane pobyty",
    "has_company": "Rezerwacja firmowa",
    "days_in_waiting_list": "Dni na liście oczekujących",
    "adr": "Cena za noc (ADR)",
    "adr_per_person": "Cena za noc na osobę",
    "hotel": "Hotel",
    "meal": "Typ wyżywienia",
    "market_segment": "Segment rynku",
    "distribution_channel": "Kanał dystrybucji",
    "reserved_room_type": "Typ pokoju",
    "deposit_type": "Typ depozytu",
    "customer_type": "Typ klienta",
    "agent_cat": "Agent / pośrednik",
}


def snapshot_mask(df: pd.DataFrame, h: int) -> pd.Series:
    """Czy rezerwacja była aktywna h dni przed przyjazdem."""
    booked = df["lead_time"] >= h
    not_yet_cancelled = (df["is_cancel"] == 0) | (df["cancel_days_before"] <= h)
    return booked & not_yet_cancelled


def build_snapshots(df: pd.DataFrame, horizons=None) -> pd.DataFrame:
    horizons = horizons or C.SNAPSHOT_HORIZONS
    parts = []
    for h in horizons:
        part = df.loc[snapshot_mask(df, h)].copy()
        part["days_to_arrival"] = h
        parts.append(part)
    snap = pd.concat(parts, ignore_index=True)
    snap["snapshot_date"] = snap["arrival"] - pd.to_timedelta(snap["days_to_arrival"], unit="D")
    return snap


def make_features(rows: pd.DataFrame, days_to_arrival=None,
                  categories: dict | None = None) -> pd.DataFrame:
    """Macierz cech dla wierszy (rezerwacji) w danym momencie przed przyjazdem.

    days_to_arrival: Series/array; jeśli None - używa kolumny rows['days_to_arrival'].
    categories: słownik poziomów kategorii z treningu (spójność przy scoringu).
    """
    X = pd.DataFrame(index=rows.index)
    dta = rows["days_to_arrival"] if days_to_arrival is None else days_to_arrival
    X["days_to_arrival"] = np.asarray(dta, dtype=float)
    X["lead_time"] = rows["lead_time"].astype(float)
    X["days_since_booking"] = X["lead_time"] - X["days_to_arrival"]
    X["arrival_dow"] = rows["arrival"].dt.dayofweek
    for col in ["stays_in_weekend_nights", "stays_in_week_nights", "nights",
                "adults", "children", "babies", "is_repeated_guest",
                "previous_cancellations", "previous_bookings_not_canceled",
                "has_company"]:
        X[col] = rows[col].astype(float)
    # Dni na liście oczekujących znane tylko do momentu migawki
    X["days_in_waiting_list"] = np.minimum(rows["days_in_waiting_list"],
                                           X["days_since_booking"].clip(lower=0))
    for col in CATEGORICAL:
        if categories is not None:
            vals = rows[col].astype(str)
            vals = vals.where(vals.isin(categories[col]))      # nieznane -> brak
            X[col] = pd.Categorical(vals, categories=categories[col])
        else:
            X[col] = rows[col].astype("category")
    return X[FEATURES]


def category_levels(rows: pd.DataFrame) -> dict:
    return {col: sorted(rows[col].astype(str).unique().tolist()) for col in CATEGORICAL}
