"""Codzienna lista rezerwacji wysokiego ryzyka z powodami i zalecanym działaniem."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda import HOTEL_PL
from src.explain import risk_reasons


def risk_list(df, model, as_of, days_ahead=14, contact_threshold=0.25, call_threshold=0.5):
    as_of = pd.Timestamp(as_of)
    dta = (df["arrival"] - as_of).dt.days
    active = (dta >= 0) & (dta <= days_ahead) & (df["lead_time"] >= dta) & (
        (df["is_cancel"] == 0) | (df["cancel_days_before"] <= dta))
    rows = df[active].copy()
    rows["dta"] = dta[active]
    rows["p"] = model.predict(rows, days_to_arrival=rows["dta"])
    rows["revenue_at_risk"] = rows["p"] * rows["revenue"]
    rows = rows.sort_values("p", ascending=False)

    def action(p):
        if p >= call_threshold:
            return "Telefon: potwierdzenie + oferta zmiany terminu / przedpłaty z rabatem"
        if not np.isnan(contact_threshold) and p >= contact_threshold:
            return "E-mail: prośba o potwierdzenie przyjazdu + oferta przedpłaty"
        return "Brak działań"
    rows["action"] = rows["p"].map(action)
    top = rows[rows["action"] != "Brak działań"].copy()
    top["reasons"] = risk_reasons(model, top, days_to_arrival=top["dta"]) if len(top) else []

    out = top.assign(hotel=top["hotel"].map(HOTEL_PL))[[
        "booking_id", "hotel", "arrival", "dta", "nights", "distribution_channel",
        "market_segment", "deposit_type", "adr", "revenue", "p", "revenue_at_risk",
        "action", "reasons", "no_arrival"]]
    out = out.rename(columns={
        "booking_id": "ID rezerwacji (pseudonim)", "hotel": "Hotel", "arrival": "Przyjazd",
        "dta": "Dni do przyjazdu", "nights": "Noce", "distribution_channel": "Kanał",
        "market_segment": "Segment", "deposit_type": "Depozyt", "adr": "ADR [EUR]",
        "revenue": "Wartość rezerwacji [EUR]", "p": "Ryzyko nieprzyjazdu",
        "revenue_at_risk": "Przychód zagrożony [EUR]", "action": "Zalecane działanie",
        "reasons": "Główne powody ryzyka", "no_arrival": "Faktycznie nie przyjechał (weryfikacja)"})
    summary = {
        "Aktywne rezerwacje (przyjazd w ciągu 14 dni)": len(rows),
        "Do kontaktu telefonicznego": int((rows["action"].str.startswith("Telefon")).sum()),
        "Do kontaktu e-mail": int((rows["action"].str.startswith("E-mail")).sum()),
        "Oczekiwana liczba nieprzyjazdów": float(rows["p"].sum()),
        "Faktyczna liczba nieprzyjazdów (weryfikacja)": int(rows["no_arrival"].sum()),
        "Przychód zagrożony [EUR]": float(rows["revenue_at_risk"].sum()),
    }
    return out.reset_index(drop=True), summary
