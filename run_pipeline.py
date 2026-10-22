"""
Uruchomienie całego projektu (PyCharm / Spyder / terminal):

    python run_pipeline.py                     # pełny przebieg + raport HTML w przeglądarce
    python run_pipeline.py --as-of 2017-07-01  # inny "dzisiejszy" dzień planu
    python run_pipeline.py --no-browser        # bez otwierania przeglądarki

Wyniki: outputs/reports/raport_klienta.html, outputs/reports/wyniki_projektu.xlsx,
        outputs/figures/*.png, outputs/tables/*.csv, outputs/models/*
"""
import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
if "ipykernel" not in sys.modules:          # w Spyder/PyCharm bez okien wykresów
    matplotlib.use("Agg")

import pandas as pd

import config as C
from src.pipeline import run_all
from src.report import write_excel, write_html


def main():
    ap = argparse.ArgumentParser(description="Anulacje i overbooking - pełny pipeline")
    ap.add_argument("--as-of", default=C.DEMO_AS_OF, help="data 'dzisiejsza' (2017-01-15 .. 2017-08-31)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    R = run_all(as_of=args.as_of)
    html_path = write_html(R)
    xlsx_path = write_excel(R)

    T = R["tables"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 12)
    print("\n=== Porównanie modeli (okres testowy 2017) ===")
    print(T["model_comparison"][["ROC_AUC", "PR_AUC", "Brier", "ECE"]].round(4).to_string())
    print("\n=== Backtest overbookingu 2017 ===")
    print(T["backtest_summary"].round(0).to_string())
    print("\n=== Business case ===")
    for k, v in T["kpi"].items():
        print(f"  {k:35s} {v:,.2f}")
    print(f"\nRaport HTML : {html_path}")
    print(f"Excel       : {xlsx_path}")
    if not args.no_browser:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
