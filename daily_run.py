"""
Codzienne uruchomienie produkcyjne: plan overbookingu + lista rezerwacji do kontaktu
+ raport HTML gotowy do publikacji online (outputs/site/index.html).

Przykłady:
    python daily_run.py                       # dzień demonstracyjny z config.DEMO_AS_OF
    python daily_run.py --as-of 2017-07-10    # konkretny dzień w danych
    python daily_run.py --replay              # tryb DEMO do harmonogramu: dzisiejsza data
                                              #   -> kolejny dzień okresu testowego 2017
    python daily_run.py --as-of today --data D:/pms/eksport_dzisiejszy.csv   # tryb produkcyjny
    python daily_run.py --replay --publish-dir \\\\serwer\\www\\raport --open

Wyniki:
    outputs/daily/<data>/      plan_overbookingu.xlsx, lista_ryzyka.xlsx, plan.png, raport_dzienny.html
    outputs/site/              index.html (najnowszy raport), archiwum/, pliki/  -> do publikacji
Jeśli modele nie istnieją, zostaną wytrenowane automatycznie (ok. 1-2 min).
"""
import argparse
import json
import shutil
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")

import joblib
import pandas as pd

import config as C
from src import actions
from src import overbooking as ob
from src.daily_report import build_daily_html, replay_date, write_site
from src.data import load_clean
from src.eda import HOTEL_PL
from src.modeling import CancellationModel
from src.occupancy import last_minute_show_rate, nightly_occupancy, pickup_matrix
from src.overbooking_plots import fig_forward_plan


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def ensure_models(data_path=None):
    if (C.MODELS / "occupancy_forecaster.joblib").exists() and (C.MODELS / "settings.json").exists():
        return
    log("Brak modeli - trenuję (jednorazowo, ok. 1-2 min) ...")
    from src.pipeline import run_all
    run_all(save_outputs=True, verbose=True)


def main():
    ap = argparse.ArgumentParser(description="Dzienny raport overbookingu i ryzyka anulacji")
    ap.add_argument("--as-of", default=None,
                    help="data raportu (RRRR-MM-DD) lub 'today'; domyślnie config.DEMO_AS_OF")
    ap.add_argument("--replay", action="store_true",
                    help="tryb DEMO: dzisiejsza data mapowana na kolejny dzień okresu testowego 2017")
    ap.add_argument("--data", default=None, help="ścieżka do aktualnego eksportu z PMS (CSV)")
    ap.add_argument("--days", type=int, default=C.FORWARD_DAYS)
    ap.add_argument("--publish-dir", default=None,
                    help="opcjonalnie: folder, do którego skopiować stronę (np. udział sieciowy serwera WWW)")
    ap.add_argument("--open", action="store_true", help="otwórz raport w przeglądarce")
    args = ap.parse_args()

    if args.replay:
        as_of = replay_date()
        mode_note = f"DEMO: dane historyczne, dzień w danych {as_of:%d.%m.%Y}"
    elif args.as_of == "today":
        as_of, mode_note = pd.Timestamp(datetime.now().date()), ""
    else:
        as_of, mode_note = pd.Timestamp(args.as_of or C.DEMO_AS_OF), "DEMO: dane historyczne"
    log(f"Raport na dzień {as_of:%Y-%m-%d}")

    ensure_models(args.data)
    model = CancellationModel.load()
    forecaster = joblib.load(C.MODELS / "occupancy_forecaster.joblib")
    settings = json.loads((C.MODELS / "settings.json").read_text(encoding="utf-8"))
    capacity = settings["capacity"]
    thr = settings["contact_threshold"] if settings["contact_threshold"] is not None else float("nan")

    try:
        df = load_clean(args.data)
    except FileNotFoundError as exc:
        sys.exit(f"BŁĄD: {exc}")
    last = df["arrival"].max()
    if as_of > last or as_of < df["arrival"].min() + pd.Timedelta(days=60):
        sys.exit(f"BŁĄD: brak danych dla dnia {as_of:%Y-%m-%d} (dane obejmują przyjazdy do {last:%Y-%m-%d}). "
                 "Użyj --replay (tryb demo) albo --data z aktualnym eksportem z PMS.")
    occ = nightly_occupancy(df)
    picks = pickup_matrix(df)
    q = last_minute_show_rate(df)

    plan = ob.forward_plan(df, model, occ, capacity, picks, q, forecaster, as_of, days=args.days)
    risk, summary = actions.risk_list(df, model, as_of, contact_threshold=thr)

    out = C.DAILY / f"{as_of:%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    plan_out = plan.assign(hotel=plan["hotel"].map(HOTEL_PL)).drop(columns=["realized"])
    plan_out.to_excel(out / "plan_overbookingu.xlsx", index=False)
    risk.drop(columns=["Faktycznie nie przyjechał (weryfikacja)"]).to_excel(out / "lista_ryzyka.xlsx", index=False)
    fig = fig_forward_plan(plan, as_of)
    fig.savefig(out / "plan.png")
    (out / "raport_dzienny.html").write_text(
        build_daily_html(plan, risk, summary, as_of, fig, mode_note,
                         downloads={"Plan overbookingu (Excel)": "plan_overbookingu.xlsx",
                                    "Lista rezerwacji do kontaktu (Excel)": "lista_ryzyka.xlsx"}),
        encoding="utf-8")
    index = write_site(plan, risk, summary, as_of, fig, out, mode_note)

    if args.publish_dir:
        dest = Path(args.publish_dir)
        shutil.copytree(C.SITE, dest, dirs_exist_ok=True)
        log(f"Strona skopiowana do: {dest}")

    log("Podsumowanie:")
    for k, v in summary.items():
        print(f"    {k:50s} {v:,.1f}" if isinstance(v, float) else f"    {k:50s} {v}")
    log(f"Raport: {index}")
    if args.open:
        webbrowser.open(index.resolve().as_uri())


if __name__ == "__main__":
    main()
