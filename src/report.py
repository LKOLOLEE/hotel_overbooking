"""Raport dla Klienta: HTML (jeden plik, wykresy osadzone) + skoroszyt Excel."""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import numpy as np
import pandas as pd

import config as C
from src.eda import HOTEL_PL
from src.overbooking import POLICY_NAMES

PCT_KEYS = ["Odsetek", "Udział", "Ryzyko nieprzyjazdu", "Pokrycie", "WAPE", "ROI", "Uplift",
            "CI dolny", "CI górny", "Śr. prognoza", "Rzeczywisty odsetek", "prognozowane ryzyko",
            "Nieprzyjazdy", "Błąd kalibracji", "% potencjału", "Śr. rzeczywista", "p_full",
            "Szansa", "ECE", "Spadek"]


# ---------------------------------------------------------------------------
# Formatowanie
# ---------------------------------------------------------------------------
def _num(x, dec=0):
    if pd.isna(x):
        return "-"
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")


def fmt_pct(x, dec=1):
    return "-" if pd.isna(x) else f"{x * 100:.{dec}f}%".replace(".", ",")


def fmt_eur(x):
    return "-" if pd.isna(x) else f"{_num(x)} €"


def format_table(df: pd.DataFrame, decimals: dict | None = None) -> pd.DataFrame:
    decimals = decimals or {}
    out = df.copy()
    for col in out.columns:
        s = out[col]
        name = str(col)
        if s.dtype == object:                      # np. kolumna z dict-a o mieszanych typach
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() == s.notna().sum() and s.notna().any():
                s = num
        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = s.dt.strftime("%d.%m.%Y")
        elif not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        elif name in decimals:
            out[col] = s.map(lambda v: _num(v, decimals[name]))
        elif "EUR" in name:
            out[col] = s.map(fmt_eur)
        elif any(k in name for k in PCT_KEYS):
            out[col] = s.map(fmt_pct)
        elif (s.dropna() % 1 == 0).all():
            out[col] = s.map(lambda v: _num(v, 0))
        else:
            out[col] = s.map(lambda v: _num(v, 2))
    return out


def table_html(df: pd.DataFrame, index=True, decimals=None, max_rows=None, transpose=False) -> str:
    d = df.head(max_rows) if max_rows else df
    f = format_table(d, decimals)
    if transpose:
        f = f.T
    return f.to_html(index=index, escape=True, border=0, classes="tbl", na_rep="-")


def fig_html(fig, caption="") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">{cap}</figure>'


def kpi_tile(label, value, note="", tone="") -> str:
    return (f'<div class="kpi {tone}"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(value)}</div>'
            f'<div class="kpi-note">{html.escape(note)}</div></div>')


CSS = """
:root{--surface:#fcfcfb;--panel:#ffffff;--text:#0b0b0b;--text2:#52514e;--line:#e4e3df;
--accent:#2a78d6;--good:#0ca30c;--bad:#d03b3b;--warn:#b07a00}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{background:#0d366b;color:#fff;padding:36px 16px}
header .wrap{max-width:1120px;margin:0 auto}
header h1{margin:0 0 6px;font-size:28px;line-height:1.2}
header p{margin:0;color:#cde2fb}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
nav{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:0 0 28px}
nav ol{margin:0;padding-left:28px;columns:2;column-gap:40px}
nav a{color:var(--accent);text-decoration:none}
section{margin:0 0 40px}
h2{font-size:21px;margin:0 0 4px;padding-top:8px;border-top:2px solid var(--text)}
h3{font-size:16px;margin:22px 0 6px}
.lead{color:var(--text2);margin:4px 0 14px}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.kpis .kpi{flex:1 1 220px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi-label{font-size:12px;color:var(--text2);text-transform:uppercase;letter-spacing:.04em}
.kpi-value{font-size:26px;font-weight:700;margin:4px 0;font-variant-numeric:tabular-nums}
.kpi-note{font-size:12px;color:var(--text2)}
.kpi.good .kpi-value{color:var(--good)} .kpi.bad .kpi-value{color:var(--bad)}
figure{margin:14px 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
figure img{max-width:100%;height:auto;display:block;margin:0 auto}
figcaption{font-size:12.5px;color:var(--text2);margin-top:6px}
.grid2{display:flex;flex-wrap:wrap;gap:12px}
.grid2>*{flex:1 1 360px;min-width:0}
@media(max-width:760px){nav ol{columns:1}}
.tblwrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:10px 0}
table.tbl{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
table.tbl th{background:#f3f2ee;text-align:left;font-weight:600;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
table.tbl td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
table.tbl tr:last-child td{border-bottom:none}
.callout{background:#eef5fd;border-left:4px solid var(--accent);padding:12px 16px;border-radius:6px;margin:12px 0}
.synth{background:#fff6e0;border-left:4px solid #eda100;padding:10px 14px;border-radius:6px;margin:12px 0;font-size:13.5px}
ul.findings li{margin:4px 0}
footer{color:var(--text2);font-size:12px;border-top:1px solid var(--line);padding-top:12px}
"""


def _tw(inner):
    return f'<div class="tblwrap">{inner}</div>'


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def build_html(R: dict) -> str:
    T, F = R["tables"], R["figures"]
    kpi, bt, comp = T["kpi"], T["backtest_summary"], T["model_comparison"]
    ov, rs, meta = T["outreach_overall"], T["risk_summary"], T["meta"]
    as_of = pd.Timestamp(meta["as_of"])
    m_name, s_name = POLICY_NAMES["model"], POLICY_NAMES["static"]
    auc_m = comp.loc["LightGBM + kalibracja (model docelowy)", "ROC_AUC"]
    auc_h = comp.loc["Heurystyka segmentowa (stan obecny)", "ROC_AUC"]
    eda = T["eda_summary"]

    sections = []

    # 1. Podsumowanie -----------------------------------------------------------
    tiles = "".join([
        kpi_tile("Korzyść netto rocznie", fmt_eur(kpi["Korzyść netto rocznie [EUR]"]),
                 "2 hotele, po kosztach utrzymania", "good"),
        kpi_tile("ROI w 1. roku", fmt_pct(kpi["ROI w 1. roku"], 0),
                 f"koszt projektu {fmt_eur(kpi['Koszt projektu [EUR]'])}"),
        kpi_tile("Okres zwrotu", f"{_num(kpi['Okres zwrotu [miesiące]'], 1)} mies.", ""),
        kpi_tile("Backtest 2017: model ML", fmt_eur(bt.loc[m_name, "Wynik netto [EUR]"]),
                 f"{fmt_pct(bt.loc[m_name, '% potencjału'], 0)} maksymalnego potencjału", "good"),
        kpi_tile("Backtest 2017: stały overbooking", fmt_eur(bt.loc[s_name, "Wynik netto [EUR]"]),
                 f"{_num(bt.loc[s_name, 'Relokacje gości'])} relokowanych gości", "bad"),
        kpi_tile("Trafność modelu (ROC AUC)", _num(auc_m, 3).replace(" ", ""),
                 f"heurystyka segmentowa: {_num(auc_h, 3)}"),
    ])
    late_city = eda["Przychód anulowany późno (<=7 dni) [EUR]"].sum()
    findings = [
        f"W analizowanych hotelach {fmt_pct(eda.loc['Hotel miejski', 'Odsetek anulacji'], 0)} rezerwacji w hotelu miejskim "
        f"i {fmt_pct(eda.loc['Resort', 'Odsetek anulacji'], 0)} w resorcie zostało anulowanych; "
        f"anulacje w ostatnim tygodniu przed przyjazdem objęły rezerwacje o wartości {fmt_eur(late_city)}.",
        f"Model ryzyka (LightGBM z kalibracją) przewiduje nieprzyjazd na poziomie pojedynczej rezerwacji "
        f"z ROC AUC {_num(auc_m, 3)} wobec {_num(auc_h, 3)} dla heurystyki segmentowej, a jego prawdopodobieństwa są skalibrowane.",
        f"W backteście 2017 (decyzja na D-{C.DECISION_HORIZON_DAYS}) rekomendacje modelu dały {fmt_eur(bt.loc[m_name, 'Wynik netto [EUR]'])} "
        f"wyniku netto przy {_num(bt.loc[m_name, 'Relokacje gości'])} relokacjach, podczas gdy stały overbooking "
        f"{int(C.STATIC_OVERBOOKING_PCT * 100)}% przyniósł {fmt_eur(bt.loc[s_name, 'Wynik netto [EUR]'])}.",
        f"Kontakt z gośćmi o ryzyku powyżej {fmt_pct(ov['Rekomendowany próg ryzyka do kontaktu'], 0)} "
        f"(ok. {fmt_pct(ov['Odsetek rezerwacji do kontaktu'], 0)} rezerwacji) jest opłacalny w symulacji (dane syntetyczne).",
        f"Na dzień {as_of:%d.%m.%Y}: {rs['Do kontaktu telefonicznego']} rezerwacji do kontaktu telefonicznego i "
        f"{rs['Do kontaktu e-mail']} do kontaktu e-mail w ciągu najbliższych 14 dni.",
    ]
    sections.append(("podsumowanie", "Podsumowanie dla zarządu",
                     f'<div class="kpis">{tiles}</div><ul class="findings">'
                     + "".join(f"<li>{html.escape(f)}</li>" for f in findings) + "</ul>"
                     + '<div class="synth">Wyniki finansowe częściowo opierają się na założeniach syntetycznych '
                       '(koszt relokacji, popyt odrzucony w noce wyprzedane, efekt kontaktu z gościem). '
                       'Pełna lista w sekcji „Założenia i ograniczenia”.</div>'))

    # 2. Diagnoza ---------------------------------------------------------------
    sections.append(("diagnoza", "Diagnoza: skala problemu anulacji",
                     '<p class="lead">Dane: publiczny zbiór Hotel Booking Demand (hotel miejski i resort, '
                     'przyjazdy 07.2015-08.2017).</p>'
                     + _tw(table_html(eda, decimals={"Liczba rezerwacji": 0, "Śr. wyprzedzenie rezerwacji (dni)": 0},
                                     transpose=True))
                     + fig_html(F["eda_trend"]) + '<div class="grid2">' + fig_html(F["eda_lead"])
                     + fig_html(F["eda_segment"]) + "</div>" + fig_html(F["eda_timing"])
                     + fig_html(F["eda_occupancy"], "Pojemność oszacowana jako maksymalne zrealizowane obłożenie.")))

    # 3. Model --------------------------------------------------------------------
    comp_t = comp.copy()
    comp_t.index.name = "Model"
    sections.append(("model", "Model ryzyka nieprzyjazdu",
                     '<p class="lead">Model ocenia ryzyko w konkretnym dniu przed przyjazdem, używając wyłącznie '
                     'informacji dostępnych w tym dniu (migawki bez wycieku danych). Walidacja czasowa: '
                     'trening 2015/16, kalibracja II poł. 2016, test 2017.</p>'
                     + _tw(table_html(comp_t, decimals={"ROC_AUC": 3, "PR_AUC": 3, "Brier": 4, "LogLoss": 4}))
                     + '<div class="grid2">' + fig_html(F["model_roc"]) + fig_html(F["model_calibration"]) + "</div>"
                     + fig_html(F["model_auc_horizon"])
                     + _tw(table_html(T["model_by_horizon"], decimals={"ROC AUC (heurystyka)": 3, "ROC AUC (LightGBM)": 3,
                                                                        "Brier (heurystyka)": 4, "Brier (LightGBM)": 4}))))

    # 4. SHAP -----------------------------------------------------------------------
    sections.append(("powody", "Dlaczego goście nie przyjeżdżają",
                     fig_html(F["shap_importance"]) + fig_html(F["shap_categories"]) + fig_html(F["shap_dependence"])
                     + '<div class="callout">Cena (ADR) oraz miesiąc przyjazdu zostały świadomie wyłączone z modelu: '
                       'przy jednym roku historii model zapamiętywał sezon konkretnego roku, a ceny dryfują w czasie, '
                       'co pogarszało wyniki na danych z 2017 r.</div>'))

    # 5. Prognoza obłożenia ---------------------------------------------------------
    sections.append(("prognoza", "Prognoza obłożenia",
                     '<p class="lead">Prognoza strukturalna (rezerwacje x ryzyko + spodziewane nowe rezerwacje) '
                     'jest korygowana modelem 2. stopnia, który uczy się praktyki sprzedażowej hotelu. '
                     'Niepewność pochodzi z reszt walidacji krzyżowej (podejście konformalne).</p>'
                     + fig_html(F["forecast_accuracy"]) + _tw(table_html(T["forecast_accuracy"],
                                                                         decimals={"Liczba nocy": 0}))))

    # 6. Backtest -----------------------------------------------------------------
    sections.append(("backtest", "Overbooking: backtest na 2017 r.",
                     f'<p class="lead">Dla każdej nocy w 2017 r. decyzja zapadała na D-{C.DECISION_HORIZON_DAYS}, '
                     'na podstawie wyłącznie ówczesnej wiedzy. Wynik to przychód z pokoi, które pozostałyby puste, '
                     'pomniejszony o koszt relokacji gości.</p>'
                     + fig_html(F["ob_example"]) + fig_html(F["ob_cumulative"]) + fig_html(F["ob_bars"])
                     + _tw(table_html(bt))
                     + '<div class="synth">Popyt odrzucony w noce wyprzedane i koszt relokacji są syntetyczne '
                       '(zbiór zawiera tylko przyjęte rezerwacje).</div>'))

    # 7. Plan -------------------------------------------------------------------
    plan = T["forward_plan"].copy()
    plan_t = plan.assign(Hotel=plan["hotel"].map(HOTEL_PL))[[
        "Hotel", "night", "horizon", "capacity", "in_house", "on_books", "forecast", "occ_p10", "occ_p90",
        "p_full", "x_opt", "exp_gain", "rekomendacja"]].rename(columns={
            "night": "Noc", "horizon": "Dni do nocy", "capacity": "Pojemność", "in_house": "Goście w hotelu",
            "on_books": "Rezerwacje na książce", "forecast": "Prognoza obłożenia", "occ_p10": "P10",
            "occ_p90": "P90", "p_full": "Szansa pełnego hotelu", "x_opt": "Dodatkowe rezerwacje",
            "exp_gain": "Oczekiwany zysk [EUR]", "rekomendacja": "Rekomendacja"})
    plan_t = plan_t.sort_values(["Noc", "Hotel"]).head(28)
    sections.append(("plan", f"Plan overbookingu od {as_of:%d.%m.%Y}",
                     fig_html(F["forward_plan"]) + "<h3>Najbliższe 14 nocy</h3>"
                     + _tw(table_html(plan_t, index=False, decimals={"Prognoza obłożenia": 0}))
                     + '<p class="lead">Pełny plan na 60 dni: arkusz „Plan_overbooking” w pliku Excel.</p>'))

    # 8. Lista ryzyka --------------------------------------------------------------
    rl = T["risk_list"].drop(columns=["Faktycznie nie przyjechał (weryfikacja)"])
    rs_t = pd.DataFrame({"Wartość": rs}).T
    sections.append(("ryzyko", "Lista rezerwacji wysokiego ryzyka (najbliższe 14 dni)",
                     _tw(table_html(rs_t, index=False))
                     + _tw(table_html(rl, index=False, max_rows=25, decimals={"Dni do przyjazdu": 0, "Noce": 0,
                                                                              "ID rezerwacji (pseudonim)": 0}))
                     + '<p class="lead">Pełna lista: arkusz „Lista_ryzyka”. Identyfikatory są pseudonimami - '
                       'model nie wymaga danych osobowych gości.</p>'))

    # 9. Działania prewencyjne ---------------------------------------------------
    ov_t = pd.DataFrame({"Wartość": {k: v for k, v in ov.items()}}).T
    sections.append(("prewencja", "Działania prewencyjne: test A/B kontaktu z gościem",
                     '<div class="synth">Ta sekcja jest SYMULACJĄ: efekt kontaktu nałożono syntetycznie na '
                     'prawdziwe rezerwacje, aby pokazać metodę oceny i wyboru grupy docelowej. '
                     'Po wdrożeniu zastępuje ją wynik prawdziwego testu.</div>'
                     + _tw(table_html(ov_t, index=False)) + fig_html(F["outreach_uplift"])
                     + _tw(table_html(T["outreach_deciles"]))))

    # 10. Polityki --------------------------------------------------------------
    wi = T["policy_whatif"]
    sections.append(("polityki", "Polityki anulacji i przedpłat",
                     fig_html(F["policy_heat"]) + fig_html(F["policy_whatif"])
                     + _tw(table_html(wi[["Hotel", "Kanał", "Wyprzedzenie (dni)", "Rezerwacje", "Odsetek anulacji",
                                          "Odsetek późnych anulacji", "Zmiana przychodu rocznie [EUR]",
                                          "Rekomendacja"]], index=False, max_rows=12))
                     + _tw(table_html(T["channels"], index=False))))

    # 11. Monitoring ------------------------------------------------------------
    sections.append(("monitoring", "Monitoring modelu",
                     '<div class="grid2">' + fig_html(F["drift"]) + _tw(table_html(T["drift"], index=False, max_rows=14,
                                                                                    decimals={"PSI": 3})) + "</div>"
                     + fig_html(F["monthly"])
                     + _tw(table_html(T["monthly"], decimals={"ROC AUC": 3, "Brier": 4}))
                     + '<p class="lead">Zasada: PSI &gt; 0,25 lub błąd kalibracji &gt; 3 pkt proc. przez 2 miesiące '
                       '→ retrening i ponowna kalibracja.</p>'))

    # 12. Business case -----------------------------------------------------------
    items = T["business_items"]
    sections.append(("business", "Business case",
                     fig_html(F["business"]) + _tw(table_html(items))
                     + f'<p class="lead">Wynik backtestu (8 miesięcy) przeliczono na rok proporcjonalnie do liczby nocy. '
                       f'W sieci hotelowej efekt skaluje się z liczbą obiektów przy niemal zerowym koszcie krańcowym '
                       f'(ten sam kod i model centralny).</p>'))

    # 13. Założenia --------------------------------------------------------------
    limits = [
        "Zbiór obejmuje 2 hotele w Portugalii (2015-2017); wyniki należy potwierdzić na danych Klienta.",
        "Liczba pokoi nie występuje w danych - przyjęto maksymalne zrealizowane obłożenie.",
        "Zbiór zawiera tylko przyjęte rezerwacje; popyt odrzucony w noce wyprzedane jest syntetyczny.",
        "Analiza na poziomie hotelu (nie typu pokoju); rozszerzenie na typy pokoi wymaga danych o pojemności typów i upgrade'ach.",
        "Zmienna 'country' oraz pola uzupełniane po fakcie zostały wyłączone (wyciek informacji).",
        "Annualizacja wyniku backtestu (styczeń-sierpień) może zawyżać efekt, bo pomija niski sezon jesienny.",
    ]
    sections.append(("zalozenia", "Założenia i ograniczenia",
                     _tw(table_html(T["assumptions"])) + "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in limits)
                     + "</ul>"))

    nav = "".join(f'<li><a href="#{sid}">{html.escape(title)}</a></li>' for sid, title, _ in sections)
    body = "".join(f'<section id="{sid}"><h2>{html.escape(title)}</h2>{content}</section>'
                   for sid, title, content in sections)
    return f"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Raport anulacji i overbookingu</title><style>{CSS}</style></head><body>
<header><div class="wrap"><h1>Prognozowanie anulacji i inteligentny overbooking</h1>
<p>Raport dla Klienta · dane: Hotel Booking Demand · stan na {as_of:%d.%m.%Y} ·
wygenerowano {datetime.now():%d.%m.%Y %H:%M}</p></div></header>
<main><nav><strong>Spis treści</strong><ol>{nav}</ol></nav>{body}
<footer>Źródło danych: Antonio, Almeida, Nunes (2019), „Hotel booking demand datasets”, Data in Brief 22, licencja CC BY 4.0.
Elementy oznaczone jako SYNTETYCZNE są założeniami do zastąpienia danymi Klienta.</footer></main></body></html>"""


def write_html(R: dict, path=None):
    path = path or C.REPORTS / "raport_klienta.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(R), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
def write_excel(R: dict, path=None):
    path = path or C.REPORTS / "wyniki_projektu.xlsx"
    T = R["tables"]
    plan = T["forward_plan"].copy()
    plan["hotel"] = plan["hotel"].map(HOTEL_PL)
    kpi = pd.DataFrame({"Wartość": T["kpi"]})
    sheets = {
        "Podsumowanie": kpi,
        "Plan_overbooking": plan,
        "Lista_ryzyka": T["risk_list"],
        "Backtest_podsumowanie": T["backtest_summary"],
        "Backtest_noce": T["backtest_nights"],
        "Model_porownanie": T["model_comparison"],
        "Model_horyzonty": T["model_by_horizon"],
        "Waznosc_cech": T["feature_importance"],
        "Trafnosc_prognoz": T["forecast_accuracy"],
        "Prewencja_decyle": T["outreach_deciles"],
        "Polityki_whatif": T["policy_whatif"],
        "Kanaly": T["channels"],
        "Monitoring_PSI": T["drift"],
        "Monitoring_miesiace": T["monthly"],
        "Business_case": T["business_items"],
        "Zalozenia": T["assumptions"],
        "Diagnoza": T["eda_summary"],
    }
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d = d.copy()
            keep_index = not isinstance(d.index, pd.RangeIndex)
            d.to_excel(xw, sheet_name=name[:31], index=keep_index)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "B2" if keep_index else "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
    return path


# ---------------------------------------------------------------------------
# Wyświetlanie w Jupyter / Spyder / PyCharm
# ---------------------------------------------------------------------------
def show(obj, decimals: dict | None = None, max_rows: int | None = 30, transpose: bool = False):
    """Wyświetla wykres lub tabelę (sformatowaną po polsku) w notatniku/konsoli IPython;
    w zwykłej konsoli Pythona drukuje tabelę tekstowo, a wykres otwiera w oknie."""
    import matplotlib.figure
    try:
        from IPython.display import display
        ipy = True
    except ImportError:  # pragma: no cover
        ipy = False
    if isinstance(obj, matplotlib.figure.Figure):
        if ipy:
            display(obj)
        else:
            obj.show()
        return
    if isinstance(obj, dict):                     # słownik KPI -> pionowa tabela
        obj = pd.DataFrame([obj], index=["Wartość"])
        transpose = True
    if isinstance(obj, pd.DataFrame):
        d = obj.head(max_rows) if max_rows else obj
        f = format_table(d, decimals)
        if transpose:
            f = f.T
        if ipy:
            display(f)
        else:
            print(f.to_string())
