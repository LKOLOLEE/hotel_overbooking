"""Obłożenie nocne, pojemność hotelu i historyczny 'pickup' (rezerwacje last-minute)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def expand_stays(df: pd.DataFrame) -> pd.DataFrame:
    """Rezerwacja -> wiersz na każdą noc pobytu (night = data nocy)."""
    rep = df.loc[df.index.repeat(df["nights"])].copy()
    rep["k"] = rep.groupby(level=0).cumcount()          # która noc pobytu
    rep["night"] = rep["arrival"] + pd.to_timedelta(rep["k"], unit="D")
    return rep.reset_index(drop=True)


def nightly_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Zrealizowane obłożenie (goście, którzy przyjechali) per hotel i noc."""
    shows = expand_stays(df[df["reservation_status"] == "Check-Out"])
    occ = (shows.groupby(["hotel", "night"])
                .agg(occupied=("booking_id", "size"), adr_mean=("adr", "mean"))
                .reset_index())
    nights = pd.date_range(C.NIGHTS_START, C.NIGHTS_END, freq="D")
    full = pd.MultiIndex.from_product([sorted(df["hotel"].unique()), nights],
                                      names=["hotel", "night"])
    occ = occ.set_index(["hotel", "night"]).reindex(full).reset_index()
    occ["occupied"] = occ["occupied"].fillna(0).astype(int)
    return occ


def estimate_capacity(occ: pd.DataFrame) -> dict:
    """Pojemność = maks. zrealizowane obłożenie (chyba że podano w config)."""
    if C.CAPACITY_OVERRIDE:
        return dict(C.CAPACITY_OVERRIDE)
    return occ.groupby("hotel")["occupied"].max().astype(int).to_dict()


def pickup_matrix(df: pd.DataFrame, max_h: int = 200) -> dict:
    """Dla każdego hotelu: macierz [noc x h] = liczba gości (którzy przyjechali),
    których rezerwacje wpłynęły w ostatnich h dniach przed daną nocą
    (czyli 'pickup' po momencie decyzji D-h)."""
    shows = expand_stays(df[df["reservation_status"] == "Check-Out"])
    shows["bdbn"] = (shows["night"] - shows["booking_date"]).dt.days.clip(upper=max_h + 1)
    out = {}
    for hotel, g in shows.groupby("hotel"):
        nights = pd.date_range(g["night"].min(), g["night"].max(), freq="D")
        idx = (g["night"] - nights[0]).dt.days.to_numpy()
        cnt = np.zeros((len(nights), max_h + 2), dtype=np.int32)
        np.add.at(cnt, (idx, g["bdbn"].to_numpy()), 1)
        cum = np.cumsum(cnt, axis=1)
        # pickup(n, h) = liczba gości z bdbn < h  -> cum[:, h-1]; pickup(n,0)=0
        pick = np.zeros((len(nights), max_h + 1), dtype=np.int32)
        pick[:, 1:] = cum[:, :max_h]
        out[hotel] = pd.DataFrame(pick, index=nights)
    return out


def pickup_forecast(pick: pd.DataFrame, night: pd.Timestamp, h: int,
                    as_of: pd.Timestamp | None = None,
                    weeks: int = C.PICKUP_LOOKBACK_WEEKS) -> float:
    """Prognoza pickup metodą historyczną: średnia z tych samych dni tygodnia
    z ostatnich `weeks` tygodni, wyłącznie z nocy już zrealizowanych przed
    momentem decyzji (bez wycieku z przyszłości)."""
    decision = as_of if as_of is not None else night - pd.Timedelta(days=h)
    h = min(h, pick.shape[1] - 1)
    vals = []
    w = 1
    while len(vals) < weeks and w < 60:
        past = night - pd.Timedelta(weeks=w)
        if past < decision and past in pick.index:
            vals.append(pick.at[past, h])
        w += 1
    return float(np.mean(vals)) if vals else 0.0


def pickup_rate_forecast(pick: pd.DataFrame, occ_s: pd.Series, capacity: int,
                         night: pd.Timestamp, h: int, as_of: pd.Timestamp | None = None,
                         weeks: int = C.PICKUP_LOOKBACK_WEEKS) -> float:
    """Prognoza pickup jako ODSETEK WOLNEJ POJEMNOŚCI, który zwykle wypełnia się
    po momencie decyzji (metoda 'constrained pickup').

    Dla nocy referencyjnych (ten sam dzień tygodnia, ostatnie `weeks` tygodni,
    tylko noce już zrealizowane przed decyzją):
        slack  = pojemność - goście z rezerwacji obecnych już w chwili D-h
        rho    = suma(pickup) / suma(slack)
    Prognoza pickup = rho * max(pojemność - oczekiwani goście z obecnej książki, 0).
    Dzięki temu w noce prawie pełne prognoza pickup maleje (hotel i tak by nie
    przyjął więcej rezerwacji), a w noce puste rośnie."""
    decision = as_of if as_of is not None else night - pd.Timedelta(days=h)
    h = min(h, pick.shape[1] - 1)
    num = den = 0.0
    used, w = 0, 1
    while used < weeks and w < 60:
        past = night - pd.Timedelta(weeks=w)
        if past < decision and past in pick.index and past in occ_s.index:
            pk = pick.at[past, h]
            slack = max(capacity - (occ_s.at[past] - pk), 1)
            num += pk
            den += slack
            used += 1
        w += 1
    return float(np.clip(num / den, 0, 1.5)) if den > 0 else 0.0


def last_minute_show_rate(df: pd.DataFrame, max_h: int = 200) -> dict:
    """q(h): odsetek przyjazdów wśród rezerwacji złożonych <= h dni przed przyjazdem.
    Używane jako prawdopodobieństwo, że DODATKOWA sprzedana rezerwacja dojdzie do skutku."""
    train = df[df["split"] != "test"]
    out = {}
    for hotel, g in train.groupby("hotel"):
        lt = g["lead_time"].to_numpy()
        show = 1 - g["no_arrival"].to_numpy()
        q = np.zeros(max_h + 1)
        for h in range(max_h + 1):
            m = lt <= h
            q[h] = show[m].mean() if m.sum() >= 30 else np.nan
        out[hotel] = pd.Series(q).ffill().bfill().to_numpy()
    return out
