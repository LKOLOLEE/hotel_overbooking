"""
Działania prewencyjne: eksperyment A/B kontaktu z gościem na D-14.  [SYNTETYCZNE]

Zbiór publiczny nie zawiera żadnych działań wobec gości, więc efekt kontaktu
(e-mail z prośbą o potwierdzenie + oferta przedpłaty z rabatem / zmiany
terminu zamiast anulacji) jest SYMULOWANY na prawdziwych rezerwacjach:

  * losowo 50% rezerwacji aktywnych na D-14 trafia do grupy testowej,
  * w grupie testowej rzeczywista anulacja zostaje "uratowana" z
    prawdopodobieństwem r(p) = MAX_EFFECT * 4 * p * (1 - p)
    (największy efekt dla gości "niezdecydowanych", p ~ 0.5; dla rezerwacji
    prawie pewnych lub prawie przesądzonych kontakt niewiele zmienia).

Analiza pokazuje, jak w prawdziwym wdrożeniu ocenić efekt (różnica odsetków,
przedziały ufności, efekt w decylach ryzyka) i wyznaczyć, kogo opłaca się
kontaktować. Po wdrożeniu syntetyczny efekt zastępują wyniki prawdziwego testu.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from sklearn.linear_model import LogisticRegression

from src.plotting import COL, eur_axis, pct_axis, titles


def simulate_experiment(te: pd.DataFrame, seed=C.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)
    e = te[te["days_to_arrival"] == C.OUTREACH_HORIZON_DAYS].copy()
    e["treated"] = (rng.random(len(e)) < C.OUTREACH_TREATMENT_SHARE).astype(int)
    p = e["p_lgbm"].to_numpy()
    effect = C.OUTREACH_MAX_EFFECT * 4 * p * (1 - p)
    effect = np.where(e["deposit_type"] == "Non Refund", effect * 0.2, effect)
    e["true_effect"] = effect                                   # tylko do kontroli
    saved = (e["treated"] == 1) & (e["no_arrival"] == 1) & (rng.random(len(e)) < effect)
    e["outcome"] = np.where(saved, 0, e["no_arrival"])
    e["risk_decile"] = pd.qcut(e["p_lgbm"].rank(method="first"), 10, labels=False) + 1
    return e


def _diff_ci(yc, yt):
    pc, pt = yc.mean(), yt.mean()
    se = np.sqrt(pc * (1 - pc) / len(yc) + pt * (1 - pt) / len(yt))
    d = pc - pt
    return d, d - 1.96 * se, d + 1.96 * se


def _uplift_model(e: pd.DataFrame) -> np.ndarray:
    """Gładki model efektu (uplift) jako funkcja ryzyka:
    logit P(nieprzyjazd) = b0 + b1*lp + t*(c0 + c1*lp),  lp = logit(p).
    Ogranicza szum pojedynczych decyli przy wyznaczaniu, kogo kontaktować."""
    p = e["p_lgbm"].clip(1e-4, 1 - 1e-4).to_numpy()
    lp = np.log(p / (1 - p))
    t = e["treated"].to_numpy()

    def design(tt):
        return np.column_stack([lp, tt, tt * lp])
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(design(t), e["outcome"])
    return lr.predict_proba(design(np.zeros_like(t)))[:, 1] - lr.predict_proba(design(np.ones_like(t)))[:, 1]


def analyze(e: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    e["uplift_model"] = _uplift_model(e)
    e["net_value_model"] = (e["uplift_model"] * e["revenue"] * (1 - C.RESALE_PROBABILITY)
                            - C.OUTREACH_COST_PER_CONTACT
                            - C.OUTREACH_DISCOUNT_TAKE_UP * C.OUTREACH_DISCOUNT_PCT * e["revenue"]
                            * (1 - e["p_lgbm"]))
    d, lo, hi = _diff_ci(e.loc[e["treated"] == 0, "outcome"], e.loc[e["treated"] == 1, "outcome"])
    overall = {"Rezerwacje w eksperymencie": len(e),
               "Odsetek nieprzyjazdów - kontrola": e.loc[e["treated"] == 0, "outcome"].mean(),
               "Odsetek nieprzyjazdów - kontakt": e.loc[e["treated"] == 1, "outcome"].mean(),
               "Spadek (pkt proc.)": d, "95% CI dolny": lo, "95% CI górny": hi}

    rows = []
    for dec, g in e.groupby("risk_decile"):
        c, t = g[g["treated"] == 0], g[g["treated"] == 1]
        up, ulo, uhi = _diff_ci(c["outcome"], t["outcome"])
        rev = g["revenue"].mean()
        up_m = g["uplift_model"].mean()
        value_saved = (g["uplift_model"] * g["revenue"]).mean() * (1 - C.RESALE_PROBABILITY)
        cost = C.OUTREACH_COST_PER_CONTACT + C.OUTREACH_DISCOUNT_TAKE_UP * C.OUTREACH_DISCOUNT_PCT * (
            g["revenue"] * (1 - g["p_lgbm"])).mean()
        rows.append({"Decyl ryzyka": dec,
                     "Śr. prognozowane ryzyko": g["p_lgbm"].mean(),
                     "Rezerwacje": len(g),
                     "Nieprzyjazdy - kontrola": c["outcome"].mean(),
                     "Nieprzyjazdy - kontakt": t["outcome"].mean(),
                     "Uplift (pkt proc.)": up, "CI dolny": ulo, "CI górny": uhi,
                     "Uplift - model wygładzony": up_m,
                     "Śr. przychód z rezerwacji [EUR]": rev,
                     "Wartość uratowana / kontakt [EUR]": value_saved,
                     "Koszt / kontakt [EUR]": cost,
                     "Wynik netto / kontakt [EUR]": value_saved - cost})
    by_dec = pd.DataFrame(rows).set_index("Decyl ryzyka")
    # próg: najniższe ryzyko, od którego (w górę) średni wynik netto kontaktu > 0
    s = e.sort_values("p_lgbm", ascending=False)
    cum = s["net_value_model"].cumsum().to_numpy()
    k = int(np.argmax(cum))
    thr = float(s["p_lgbm"].iloc[k]) if cum[k] > 0 else np.nan
    overall["Rekomendowany próg ryzyka do kontaktu"] = thr
    overall["Decyle do kontaktu"] = ", ".join(
        str(i) for i in by_dec.index if not np.isnan(thr) and by_dec.loc[i, "Śr. prognozowane ryzyko"] >= thr)
    overall["Odsetek rezerwacji do kontaktu"] = float((e["p_lgbm"] >= thr).mean()) if not np.isnan(thr) else 0.0
    return overall, by_dec


def annual_value(e, overall, years) -> float:
    """Roczna wartość netto kontaktowania rezerwacji powyżej progu ryzyka."""
    thr = overall["Rekomendowany próg ryzyka do kontaktu"]
    if np.isnan(thr):
        return 0.0
    return float(e.loc[e["p_lgbm"] >= thr, "net_value_model"].sum() / years)


def fig_uplift(by_dec):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = by_dec.index.to_numpy()
    ax = axes[0]
    ax.bar(x, by_dec["Uplift (pkt proc.)"], color=COL["s1"], width=0.65)
    ax.errorbar(x, by_dec["Uplift (pkt proc.)"],
                yerr=[by_dec["Uplift (pkt proc.)"] - by_dec["CI dolny"],
                      by_dec["CI górny"] - by_dec["Uplift (pkt proc.)"]],
                fmt="none", ecolor=COL["text2"], elinewidth=1, capsize=3)
    ax.plot(x, by_dec["Uplift - model wygładzony"], marker="o", ms=5, color=COL["s2"],
            label="Model wygładzony")
    ax.axhline(0, color=COL["neutral"], lw=1)
    ax.legend(loc="upper left")
    pct_axis(ax, decimals=0)
    ax.set_xticks(x)
    ax.set_xlabel("Decyl prognozowanego ryzyka (10 = najwyższe)")
    ax.grid(axis="x", visible=False)
    titles(ax, "Efekt kontaktu w decylach ryzyka", "Spadek odsetka nieprzyjazdów z 95% CI")
    ax = axes[1]
    v = by_dec["Wynik netto / kontakt [EUR]"]
    ax.bar(x, v, color=[COL["good"] if a > 0 else COL["critical"] for a in v], width=0.65)
    ax.axhline(0, color=COL["neutral"], lw=1)
    eur_axis(ax)
    ax.set_xticks(x)
    ax.set_xlabel("Decyl prognozowanego ryzyka")
    ax.grid(axis="x", visible=False)
    titles(ax, "Kogo opłaca się kontaktować", "Wynik netto na jeden kontakt [EUR]")
    fig.text(0.01, -0.03, "Dane eksperymentu są SYNTETYCZNE (symulacja na prawdziwych rezerwacjach).",
             fontsize=8, color=COL["text2"])
    fig.tight_layout()
    return fig
