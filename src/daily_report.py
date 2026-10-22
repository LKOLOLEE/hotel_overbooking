"""Dzienny raport HTML dla Klienta + statyczna strona (index + archiwum) do publikacji online."""
from __future__ import annotations

import html
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

import config as C
from src.eda import HOTEL_PL
from src.report import CSS, _tw, fig_html, fmt_eur, kpi_tile, table_html

EXTRA_CSS = """
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;background:#fff6e0;
color:#6b4a00;border:1px solid #eda100;margin-left:8px;vertical-align:middle}
.downloads a{display:inline-block;margin:4px 12px 4px 0;color:var(--accent)}
ul.archive{columns:3;font-size:13.5px} ul.archive a{color:var(--accent);text-decoration:none}
@media(max-width:760px){ul.archive{columns:1}}
"""


def replay_date(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Mapuje prawdziwą datę na dzień okresu testowego (tryb demonstracyjny)."""
    today = pd.Timestamp(today or datetime.now().date())
    start, end = pd.Timestamp(C.REPLAY_START), pd.Timestamp(C.REPLAY_END)
    span = (end - start).days + 1
    offset = (today - pd.Timestamp(C.REPLAY_ANCHOR)).days % span
    return start + pd.Timedelta(days=offset)


def _plan_table(plan: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    as_of = plan["night"].min()
    p = plan[plan["night"] < as_of + pd.Timedelta(days=days)]
    return (p.assign(Hotel=p["hotel"].map(HOTEL_PL))
             [["night", "Hotel", "capacity", "in_house", "on_books", "forecast",
               "occ_p10", "occ_p90", "p_full", "x_opt", "exp_gain", "rekomendacja"]]
             .rename(columns={"night": "Noc", "capacity": "Pojemność", "in_house": "Goście w hotelu",
                              "on_books": "Rezerwacje na książce", "forecast": "Prognoza obłożenia",
                              "occ_p10": "P10", "occ_p90": "P90", "p_full": "Szansa pełnego hotelu",
                              "x_opt": "Dodatkowe rezerwacje", "exp_gain": "Oczekiwany zysk [EUR]",
                              "rekomendacja": "Rekomendacja"})
             .sort_values(["Noc", "Hotel"]))


def build_daily_html(plan, risk, summary, as_of, fig, mode_note: str = "",
                     archive: list[str] | None = None, downloads: dict | None = None) -> str:
    as_of = pd.Timestamp(as_of)
    week = plan[plan["night"] < as_of + pd.Timedelta(days=7)]
    ob_nights = int(plan["rekomendacja"].str.startswith("Dopuść").sum())
    low_nights = int(plan["rekomendacja"].str.startswith("Niskie").sum())
    occ_week = (week["forecast"].sum() / week["capacity"].sum()) if len(week) else float("nan")
    tiles = "".join([
        kpi_tile("Prognozowane obłożenie (7 dni)", f"{occ_week:.0%}".replace(".", ","),
                 "oba hotele, prognoza na dziś"),
        kpi_tile("Noce z dopuszczalnym overbookingiem", str(ob_nights),
                 f"z {len(plan)} (hotel × noc, {plan['night'].nunique()} dni)"),
        kpi_tile("Noce z niskim obłożeniem", str(low_nights), "kandydaci do promocji", "bad" if low_nights else ""),
        kpi_tile("Rezerwacje do kontaktu", str(summary["Do kontaktu telefonicznego"] + summary["Do kontaktu e-mail"]),
                 f"tel.: {summary['Do kontaktu telefonicznego']}, e-mail: {summary['Do kontaktu e-mail']}"),
        kpi_tile("Przychód zagrożony (14 dni)", fmt_eur(summary["Przychód zagrożony [EUR]"]),
                 f"oczekiwane nieprzyjazdy: {summary['Oczekiwana liczba nieprzyjazdów']:.0f}"),
    ])
    web_cols = ["ID rezerwacji (pseudonim)", "Hotel", "Przyjazd", "Dni do przyjazdu", "Kanał",
                "Ryzyko nieprzyjazdu", "Przychód zagrożony [EUR]", "Zalecane działanie", "Główne powody ryzyka"]
    rl = risk[[c for c in web_cols if c in risk.columns]]          # pełne kolumny są w Excelu
    dl = ""
    if downloads:
        dl = '<p class="downloads">' + "".join(
            f'<a href="{html.escape(href)}" download>{html.escape(label)}</a>' for label, href in downloads.items()) + "</p>"
    arch = ""
    if archive:
        arch = ('<section id="archiwum"><h2>Archiwum raportów</h2><ul class="archive">'
                + "".join(f'<li><a href="archiwum/{d}.html">{pd.Timestamp(d):%d.%m.%Y}</a></li>' for d in archive)
                + "</ul></section>")
    badge = f'<span class="badge">{html.escape(mode_note)}</span>' if mode_note else ""
    body = f"""
<section><h2>Podsumowanie dnia</h2><div class="kpis">{tiles}</div>{dl}</section>
<section><h2>Plan overbookingu</h2>{fig_html(fig)}
<h3>Najbliższe 14 nocy</h3>{_tw(table_html(_plan_table(plan), index=False, decimals={"Prognoza obłożenia": 0}))}</section>
<section><h2>Rezerwacje do kontaktu (przyjazd w ciągu 14 dni)</h2>
{_tw(table_html(rl, index=False, max_rows=40, decimals={"ID rezerwacji (pseudonim)": 0, "Dni do przyjazdu": 0, "Noce": 0}))}
<p class="lead">Pełna lista w pliku Excel. Identyfikatory są pseudonimami.</p></section>
{arch}"""
    return f"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Raport dzienny overbooking</title><style>{CSS}{EXTRA_CSS}</style></head><body>
<header><div class="wrap"><h1>Raport dzienny: overbooking i ryzyko anulacji {badge}</h1>
<p>Stan na {as_of:%d.%m.%Y} · wygenerowano {datetime.now():%d.%m.%Y %H:%M}</p></div></header>
<main>{body}<footer>Model: LightGBM + kalibracja, prognoza obłożenia 2. stopnia. Szczegóły metodyki w raporcie projektowym.</footer>
</main></body></html>"""


def write_site(plan, risk, summary, as_of, fig, daily_dir: Path, mode_note: str = "",
               site_dir: Path = C.SITE) -> Path:
    """Zapisuje outputs/site/index.html (najnowszy raport), archiwum i pliki do pobrania."""
    as_of = pd.Timestamp(as_of)
    stamp = f"{as_of:%Y-%m-%d}"
    (site_dir / "archiwum").mkdir(parents=True, exist_ok=True)
    (site_dir / "pliki").mkdir(parents=True, exist_ok=True)
    for name in ["plan_overbookingu.xlsx", "lista_ryzyka.xlsx"]:
        shutil.copy2(daily_dir / name, site_dir / "pliki" / f"{stamp}_{name}")
    downloads = {"Plan overbookingu (Excel)": f"pliki/{stamp}_plan_overbookingu.xlsx",
                 "Lista rezerwacji do kontaktu (Excel)": f"pliki/{stamp}_lista_ryzyka.xlsx"}

    # strona archiwalna (linki względne z podfolderu)
    arch_dl = {k: "../" + v for k, v in downloads.items()}
    (site_dir / "archiwum" / f"{stamp}.html").write_text(
        build_daily_html(plan, risk, summary, as_of, fig, mode_note, downloads=arch_dl), encoding="utf-8")

    archive = sorted((p.stem for p in (site_dir / "archiwum").glob("*.html")), reverse=True)[:C.ARCHIVE_DAYS]
    index = site_dir / "index.html"
    index.write_text(build_daily_html(plan, risk, summary, as_of, fig, mode_note,
                                      archive=archive, downloads=downloads), encoding="utf-8")
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")       # GitHub Pages: serwuj pliki bez zmian
    return index
