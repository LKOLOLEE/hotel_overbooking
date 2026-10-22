"""Orkiestracja całego projektu: dane -> model -> overbooking -> działania -> raport."""
from __future__ import annotations

import json
import time

import joblib

import matplotlib.pyplot as plt
import pandas as pd

import config as C
from src import actions, business_case, eda, explain, monitoring, outreach, policy
from src import overbooking as ob
from src import overbooking_plots as obp
from src.data import load_clean
from src.features import build_snapshots
from src.modeling import train_and_evaluate
from src.occupancy import (estimate_capacity, last_minute_show_rate,
                           nightly_occupancy, pickup_matrix)
from src.plotting import save


def assumptions_table() -> pd.DataFrame:
    rows = [
        ("Pojemność hotelu", "maks. zrealizowane obłożenie" if not C.CAPACITY_OVERRIDE else str(C.CAPACITY_OVERRIDE),
         "Szacunek z danych (zbiór nie zawiera liczby pokoi)"),
        ("Koszt relokacji gościa", f"{C.WALK_COST_ADR_MULTIPLIER} x ADR + {C.WALK_COST_FIXED_EUR:.0f} EUR", "SYNTETYCZNE"),
        ("Wydatki dodatkowe na pokój-noc", str(C.ANCILLARY_PER_ROOM_NIGHT), "SYNTETYCZNE"),
        ("Noc wyprzedana (jest popyt odrzucony)", f">= {C.SOLD_OUT_THRESHOLD:.0%} pojemności", "SYNTETYCZNE"),
        ("Popyt odrzucony w noc wyprzedaną", f"Poisson({C.EXTRA_DEMAND_LAMBDA:.0%} pojemności)", "SYNTETYCZNE"),
        ("Moment decyzji o overbookingu", f"D-{C.DECISION_HORIZON_DAYS}", "Parametr"),
        ("Polityka porównawcza", f"stały overbooking {C.STATIC_OVERBOOKING_PCT:.0%}", "Parametr"),
        ("Maks. efekt kontaktu z gościem", f"{C.OUTREACH_MAX_EFFECT:.0%} mniej anulacji", "SYNTETYCZNE"),
        ("Koszt kontaktu", f"{C.OUTREACH_COST_PER_CONTACT:.0f} EUR", "SYNTETYCZNE"),
        ("Rabat za przedpłatę / odsetek przyjmujących",
         f"{C.OUTREACH_DISCOUNT_PCT:.0%} / {C.OUTREACH_DISCOUNT_TAKE_UP:.0%}", "SYNTETYCZNE"),
        ("Szansa odsprzedaży zwolnionego pokoju", f"{C.RESALE_PROBABILITY:.0%}", "SYNTETYCZNE"),
        ("Efekt wymogu przedpłaty", f"anulacje -{C.POLICY_CANCEL_REDUCTION:.0%}, rezerwacje -{C.POLICY_BOOKING_LOSS:.0%}", "SYNTETYCZNE"),
        ("Odsprzedaż wcześnie zwolnionego pokoju / zastąpienie utraconej rezerwacji",
         f"{C.POLICY_EARLY_RESALE:.0%} / {C.POLICY_REPLACEMENT:.0%}", "SYNTETYCZNE"),
        ("Koszt projektu / utrzymania", f"{C.PROJECT_COST_EUR:,.0f} / {C.RUN_COST_EUR_PER_YEAR:,.0f} EUR rocznie", "SYNTETYCZNE"),
        ("Podział czasowy", f"trening do {C.TRAIN_END}, walidacja do {C.VALID_END}, test 2017-01..08", "Metodyka"),
    ]
    return pd.DataFrame(rows, columns=["Założenie", "Wartość", "Źródło"]).set_index("Założenie")


def run_all(as_of: str = C.DEMO_AS_OF, save_outputs: bool = True, verbose: bool = True) -> dict:
    T, F = {}, {}
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}")

    for d in (C.FIGURES, C.TABLES, C.MODELS, C.REPORTS):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Dane --------------------------------------------------------------
    log("1/9 Wczytanie i czyszczenie danych")
    df = load_clean()
    years_all = (df["arrival"].max() - df["arrival"].min()).days / 365
    T["eda_summary"] = eda.summary_table(df)
    F["eda_trend"] = eda.fig_cancel_trend(df)
    F["eda_lead"] = eda.fig_cancel_by_lead(df)
    F["eda_segment"] = eda.fig_cancel_by_segment(df)
    F["eda_timing"] = eda.fig_cancel_timing(df)

    occ = nightly_occupancy(df)
    capacity = estimate_capacity(occ)
    F["eda_occupancy"] = eda.fig_occupancy(occ, capacity)

    # 2. Migawki i model -----------------------------------------------------
    log("2/9 Migawki bez wycieku danych + trening modeli")
    snap = build_snapshots(df)
    model, te, comp, by_h = train_and_evaluate(snap, verbose=verbose)
    model.save()
    T["model_comparison"], T["model_by_horizon"] = comp, by_h
    F["model_roc"] = explain.fig_roc(te)
    F["model_calibration"] = explain.fig_calibration(te)
    F["model_auc_horizon"] = explain.fig_auc_by_horizon(by_h)

    # 3. Wyjaśnialność -------------------------------------------------------
    log("3/9 Wyjaśnialność (SHAP)")
    _, Xs, contrib = explain.shap_sample(model, te)
    T["feature_importance"] = explain.importance_table(contrib)
    F["shap_importance"] = explain.fig_importance(contrib)
    F["shap_dependence"] = explain.fig_dependence(Xs, contrib)
    F["shap_categories"] = explain.fig_category_effects(Xs, contrib)

    # 4. Prognoza obłożenia ----------------------------------------------------
    log("4/9 Prognoza obłożenia (model 2. stopnia)")
    picks = pickup_matrix(df)
    q_curves = last_minute_show_rate(df)
    forecaster, _ = ob.fit_forecaster(df, model, occ, capacity, picks)
    acc, _ = ob.forecast_accuracy(df, model, occ, capacity, picks, forecaster)
    T["forecast_accuracy"] = acc
    T["forecast_cv"] = forecaster.cv_table_
    F["forecast_accuracy"] = obp.fig_forecast_accuracy(acc)

    # 5. Backtest overbookingu --------------------------------------------------
    log("5/9 Backtest polityk overbookingu (2017)")
    dec, bt = ob.backtest(df, model, occ, capacity, picks, q_curves, forecaster)
    T["backtest_summary"], T["backtest_nights"] = bt, dec
    F["ob_example"] = obp.fig_example_night(dec, forecaster)
    F["ob_cumulative"] = obp.fig_backtest_cumulative(dec)
    F["ob_bars"] = obp.fig_backtest_bars(bt)

    # 6. Działania prewencyjne (syntetyczny A/B) -------------------------------
    log("6/9 Eksperyment A/B działań prewencyjnych [syntetyczny]")
    exp = outreach.simulate_experiment(te)
    ov, by_dec = outreach.analyze(exp)
    test_years = exp["arrival"].dt.normalize().nunique() / 365
    outreach_annual = outreach.annual_value(exp, ov, test_years)
    T["outreach_overall"], T["outreach_deciles"] = ov, by_dec
    F["outreach_uplift"] = outreach.fig_uplift(by_dec)

    # 7. Polityki anulacji ------------------------------------------------------
    log("7/9 Analiza segmentów i polityk")
    seg = policy.segment_table(df)
    wi = policy.whatif_prepayment(seg, years_all)
    T["policy_whatif"], T["channels"] = wi, policy.channel_table(df, years_all)
    policy_annual = float(wi.loc[wi["Zmiana przychodu rocznie [EUR]"] > 0, "Zmiana przychodu rocznie [EUR]"].sum())
    F["policy_heat"] = policy.fig_channel_heat(df)
    F["policy_whatif"] = policy.fig_whatif(wi)

    # 8. Tryb produkcyjny: plan i lista ryzyka -----------------------------------
    log(f"8/9 Plan overbookingu i lista ryzyka na dzień {as_of}")
    plan = ob.forward_plan(df, model, occ, capacity, picks, q_curves, forecaster, as_of)
    thr = ov.get("Rekomendowany próg ryzyka do kontaktu", 0.25)
    risk, risk_sum = actions.risk_list(df, model, as_of, contact_threshold=thr)
    T["forward_plan"], T["risk_list"], T["risk_summary"] = plan, risk, risk_sum
    joblib.dump(forecaster, C.MODELS / "occupancy_forecaster.joblib")
    (C.MODELS / "settings.json").write_text(json.dumps({
        "contact_threshold": None if pd.isna(thr) else float(thr),
        "capacity": {k: int(v) for k, v in capacity.items()},
    }, indent=1), encoding="utf-8")
    F["forward_plan"] = obp.fig_forward_plan(plan, as_of)

    # 9. Monitoring i business case ------------------------------------------------
    log("9/9 Monitoring i business case")
    T["drift"] = monitoring.drift_table(model, snap)
    T["monthly"] = monitoring.monthly_performance(te)
    F["drift"] = monitoring.fig_drift(T["drift"])
    F["monthly"] = monitoring.fig_monthly(T["monthly"])
    items, kpi = business_case.build(bt, dec["night"].nunique(), outreach_annual, policy_annual)
    T["business_items"], T["kpi"] = items, kpi
    F["business"] = business_case.fig_waterfall(items)
    T["assumptions"] = assumptions_table()
    T["meta"] = {"as_of": as_of, "capacity": capacity, "n_bookings": len(df),
                 "n_snapshots": len(snap), "best_iteration": model.booster.best_iteration}

    if save_outputs:
        for name, fig in F.items():
            save(fig, name)
        for name in ["eda_summary", "model_comparison", "model_by_horizon", "feature_importance",
                     "forecast_accuracy", "backtest_summary", "backtest_nights", "outreach_deciles",
                     "policy_whatif", "channels", "forward_plan", "risk_list", "drift", "monthly",
                     "business_items", "assumptions"]:
            T[name].to_csv(C.TABLES / f"{name}.csv", encoding="utf-8-sig")
    for fig in F.values():
        plt.close(fig)
    log("Gotowe.")
    return {"tables": T, "figures": F, "model": model, "forecaster": forecaster,
            "test_snapshots": te, "experiment": exp, "snapshots": snap,
            "df": df, "occ": occ, "capacity": capacity, "picks": picks, "q_curves": q_curves}
