"""Monitoring modelu: dryf danych (PSI) i stabilność jakości w czasie."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

import config as C
from src.data import CATEGORICAL
from src.features import FEATURES, FEATURE_LABELS_PL
from src.plotting import COL, pct_axis, titles


def psi(expected: pd.Series, actual: pd.Series, categorical=False, bins=10) -> float:
    eps = 1e-4
    if categorical:
        e = expected.astype(str).value_counts(normalize=True)
        a = actual.astype(str).value_counts(normalize=True)
        idx = e.index.union(a.index)
        e, a = e.reindex(idx, fill_value=0) + eps, a.reindex(idx, fill_value=0) + eps
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.Series(np.histogram(expected, edges)[0] / len(expected)) + eps
        a = pd.Series(np.histogram(actual, edges)[0] / len(actual)) + eps
    return float(((a - e) * np.log(a / e)).sum())


def drift_table(model, snap: pd.DataFrame) -> pd.DataFrame:
    tr = model.features(snap[snap["split"] == "train"])
    te = model.features(snap[snap["split"] == "test"])
    rows = []
    for f in FEATURES:
        v = psi(tr[f], te[f], categorical=f in CATEGORICAL)
        status = "ALERT" if v >= C.PSI_ALERT else ("Uwaga" if v >= C.PSI_WARN else "OK")
        rows.append({"Cecha": FEATURE_LABELS_PL.get(f, f), "PSI": v, "Status": status})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def monthly_performance(te: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in te.groupby(te["arrival"].dt.to_period("M")):
        rows.append({"Miesiąc przyjazdu": str(m), "Migawki": len(g),
                     "ROC AUC": roc_auc_score(g["no_arrival"], g["p_lgbm"]),
                     "Brier": brier_score_loss(g["no_arrival"], g["p_lgbm"]),
                     "Śr. prognoza": g["p_lgbm"].mean(),
                     "Rzeczywisty odsetek": g["no_arrival"].mean()})
    t = pd.DataFrame(rows).set_index("Miesiąc przyjazdu")
    t["Błąd kalibracji"] = t["Śr. prognoza"] - t["Rzeczywisty odsetek"]
    t["Status"] = np.where(t["Błąd kalibracji"].abs() > 0.03, "Uwaga - rekalibracja", "OK")
    return t


def fig_drift(drift, top=12):
    d = drift.head(top).iloc[::-1]
    colors = [COL["critical"] if s == "ALERT" else COL["warning"] if s == "Uwaga" else COL["s1"]
              for s in d["Status"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(d["Cecha"], d["PSI"], color=colors, height=0.6)
    ax.axvline(C.PSI_WARN, color=COL["warning"], lw=1, ls="--")
    ax.axvline(C.PSI_ALERT, color=COL["critical"], lw=1, ls="--")
    ax.text(C.PSI_WARN, len(d) - 0.4, " uwaga", fontsize=8, color=COL["text2"])
    ax.text(C.PSI_ALERT, len(d) - 0.4, " alert", fontsize=8, color=COL["text2"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PSI (trening vs okres testowy)")
    titles(ax, "Dryf danych wejściowych", "Population Stability Index - im wyżej, tym bardziej zmieniły się dane")
    return fig


def fig_monthly(perf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    x = np.arange(len(perf))
    ax = axes[0]
    ax.plot(x, perf["ROC AUC"], marker="o", ms=5, color=COL["s1"])
    ax.set_xticks(x, perf.index, rotation=45)
    ax.set_ylim(0.5, 1)
    titles(ax, "ROC AUC w kolejnych miesiącach")
    ax = axes[1]
    ax.plot(x, perf["Śr. prognoza"], marker="o", ms=5, color=COL["s1"], label="Śr. prognoza")
    ax.plot(x, perf["Rzeczywisty odsetek"], marker="o", ms=5, color=COL["s2"], label="Rzeczywisty odsetek")
    ax.set_xticks(x, perf.index, rotation=45)
    pct_axis(ax)
    ax.set_ylim(0, None)
    ax.legend(loc="lower left")
    titles(ax, "Kalibracja w kolejnych miesiącach")
    fig.tight_layout()
    return fig
