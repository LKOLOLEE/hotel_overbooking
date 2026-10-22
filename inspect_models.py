"""
Przegląd zbudowanych modeli: LightGBM (+ kalibracja), regresja logistyczna,
heurystyka segmentowa i model obłożenia 2. stopnia.

Uruchamianie:
  * PyCharm / Spyder / VS Code: komórki "# %%" po kolei (Shift+Enter / "Run Cell"),
    wykresy w panelu Plots / SciView, tabele w konsoli.
  * Terminal / PyCharm "Run": python inspect_models.py  (wykresy zapisze do outputs/inspect/)

Jeśli brakuje któregoś modelu (np. regresji logistycznej z wcześniejszej wersji projektu),
zostanie on wytrenowany i zapisany automatycznie.
"""
# %% 0. Ustawienia
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))

import matplotlib
if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    matplotlib.use("Agg")                       # uruchomienie jako skrypt: zapis do plików
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

import config as C
from src.data import CATEGORICAL, load_clean
from src.features import FEATURE_LABELS_PL, FEATURES, build_snapshots, make_features
from src.modeling import (CancellationModel, SegmentRateModel, _lr_frame, metrics,
                          train_logistic)
from src.plotting import COL, titles

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)
OUT = C.OUTPUTS / "inspect"
OUT.mkdir(parents=True, exist_ok=True)


def show_fig(fig, name):
    fig.savefig(OUT / f"{name}.png")
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()


def header(txt):
    print("\n" + "=" * 100 + f"\n{txt}\n" + "=" * 100)


# %% 1. Wczytanie modeli i danych testowych
if not (C.MODELS / "lgbm_model.txt").exists():
    sys.exit("Brak modeli - uruchom najpierw: python run_pipeline.py")

model = CancellationModel.load()                       # LightGBM + kalibrator
df = load_clean()
snap = build_snapshots(df)
tr = snap[snap["split"] == "train"]
va = snap[snap["split"] == "valid"]
te = snap[snap["split"] == "test"].copy()

lr_path, seg_path = C.MODELS / "logistic_regression.joblib", C.MODELS / "segment_heuristic.joblib"
if not lr_path.exists() or not seg_path.exists():
    print("Brak zapisanej regresji logistycznej / heurystyki - trenuję (ok. 10-20 s) ...")
    joblib.dump(train_logistic(make_features(tr), tr["no_arrival"]), lr_path)
    joblib.dump(SegmentRateModel().fit(pd.concat([tr, va])), seg_path)
lr = joblib.load(lr_path)
seg = joblib.load(seg_path)

te["p_lgbm"] = model.predict(te)
te["p_lgbm_raw"] = model.predict_raw(te)
te["p_logistic"] = lr.predict_proba(_lr_frame(make_features(te)))[:, 1]
te["p_segment"] = seg.predict(te)

header("PLIKI MODELI (outputs/models)")
for f in sorted(C.MODELS.iterdir()):
    print(f"  {f.name:35s} {f.stat().st_size / 1024:8.1f} KB")

header("PORÓWNANIE NA OKRESIE TESTOWYM 2017")
print(pd.DataFrame({
    "Heurystyka segmentowa": metrics(te["no_arrival"], te["p_segment"]),
    "Regresja logistyczna": metrics(te["no_arrival"], te["p_logistic"]),
    "LightGBM (surowy)": metrics(te["no_arrival"], te["p_lgbm_raw"]),
    "LightGBM + kalibracja": metrics(te["no_arrival"], te["p_lgbm"]),
}).T.round(4))

# %% 2. LightGBM - parametry i struktura
b = model.booster
header("LIGHTGBM - PARAMETRY I STRUKTURA")
params = {k: v for k, v in b.params.items() if k in C.LGBM_PARAMS or k in ("num_iterations",)}
print("Parametry treningu:", {**C.LGBM_PARAMS, **params})
trees = b.trees_to_dataframe()
print(f"Liczba drzew (best_iteration): {b.num_trees()}")
print(f"Liczba cech: {b.num_feature()}  ->  {b.feature_name()}")
per_tree = trees.groupby("tree_index").agg(liscie=("node_depth", lambda d: (trees.loc[d.index, 'left_child'].isna()).sum()),
                                           glebokosc=("node_depth", "max"))
print(f"Liście na drzewo: średnio {per_tree['liscie'].mean():.1f} (num_leaves={C.LGBM_PARAMS['num_leaves']}), "
      f"maks. głębokość drzewa: {per_tree['glebokosc'].max() - 1}")
print("Kategorie zakodowane w modelu (z treningu):")
for c in CATEGORICAL:
    lv = model.categories[c]
    print(f"  {c:22s} ({len(lv):2d}): {', '.join(lv[:12])}{' ...' if len(lv) > 12 else ''}")

# %% 3. LightGBM - ważność cech (gain vs split)
imp = pd.DataFrame({
    "Cecha": [FEATURE_LABELS_PL.get(f, f) for f in b.feature_name()],
    "Gain": b.feature_importance("gain"),
    "Split": b.feature_importance("split"),
}, index=b.feature_name())
imp["Gain %"] = imp["Gain"] / imp["Gain"].sum()
imp["Split %"] = imp["Split"] / imp["Split"].sum()
imp = imp.sort_values("Gain", ascending=False)
header("LIGHTGBM - WAŻNOŚĆ CECH")
print(imp[["Cecha", "Gain %", "Split %", "Split"]].round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
top = imp.head(15).iloc[::-1]
axes[0].barh(top["Cecha"], top["Gain %"], color=COL["s1"], height=0.65)
axes[1].barh(top["Cecha"], top["Split %"], color=COL["s2"], height=0.65)
titles(axes[0], "Gain", "Udział w łącznej redukcji straty")
titles(axes[1], "Split", "Udział w liczbie podziałów w drzewach")
for a in axes:
    a.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:g}%"))
    a.grid(axis="y", visible=False)
fig.tight_layout()
show_fig(fig, "lgbm_importance")

# %% 4. LightGBM - pierwsze drzewo w formie tekstowej (bez graphviz)
def print_tree(trees, tree_index=0, max_depth=3):
    t = trees[trees["tree_index"] == tree_index].set_index("node_index")

    def rec(node, depth, prefix):
        r = t.loc[node]
        if pd.isna(r["left_child"]):
            print(f"{prefix}liść: log-odds {r['value']:+.3f}  (n={int(r['count'])})")
            return
        if depth >= max_depth:
            print(f"{prefix}... (poddrzewo, n={int(r['count'])})")
            return
        thr, feat = r["threshold"], r["split_feature"]
        if r["decision_type"] == "==":                 # podział kategoryczny: kody -> nazwy
            levels = model.categories[feat]
            names = [levels[int(k)] for k in str(thr).split("||") if int(k) < len(levels)]
            cond = f"{feat} ∈ {{{', '.join(names[:6])}{', ...' if len(names) > 6 else ''}}}"
        else:
            cond = f"{feat} {r['decision_type']} {0 if abs(float(thr)) < 1e-9 else float(thr):.4g}"
        print(f"{prefix}[{cond}]  gain={r['split_gain']:.0f}, n={int(r['count'])}")
        rec(r["left_child"], depth + 1, prefix + "   tak → ")
        rec(r["right_child"], depth + 1, prefix + "   nie → ")
    rec(f"{tree_index}-S0", 0, "")


header("LIGHTGBM - DRZEWO 0 (3 pierwsze poziomy)")
print_tree(trees, 0, max_depth=3)
print("\nPełna struktura wszystkich drzew: zmienna 'trees' (DataFrame) lub plik outputs/models/lgbm_model.txt")
trees.to_csv(OUT / "lgbm_trees.csv", index=False)
# Wersja graficzna (wymaga: pip install graphviz + program Graphviz w PATH):
# import lightgbm as lgb; lgb.plot_tree(b, tree_index=0, figsize=(30, 12), show_info=["split_gain", "leaf_count"]); plt.show()

# %% 5. Kalibracja: jak surowy wynik LightGBM zamieniany jest na prawdopodobieństwo
cal = model.calibrator
header("KALIBRATOR (regresja izotoniczna)")
print(f"Liczba progów funkcji schodkowej: {len(cal.X_thresholds_)}")
grid = np.linspace(0, 1, 11)
print(pd.DataFrame({"Surowy wynik LightGBM": grid, "Po kalibracji": cal.predict(grid)}).round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(5.5, 5))
xs = np.linspace(0, 1, 400)
ax.plot(xs, cal.predict(xs), color=COL["s1"], label="Funkcja kalibracji")
ax.plot([0, 1], [0, 1], color=COL["grid"], ls="--", lw=1, label="Brak korekty")
ax.set_xlabel("Surowe prawdopodobieństwo LightGBM")
ax.set_ylabel("Skalibrowane prawdopodobieństwo")
ax.legend(loc="upper left")
titles(ax, "Funkcja kalibracji", "Uczona na okresie walidacyjnym 07-12.2016")
show_fig(fig, "calibrator")

# %% 6. Regresja logistyczna - współczynniki i ilorazy szans
pre, clf = lr.named_steps["pre"], lr.named_steps["lr"]
names = pre.get_feature_names_out()
coef = pd.DataFrame({"Zmienna": names, "Współczynnik": clf.coef_[0]})
coef["Iloraz szans (OR)"] = np.exp(coef["Współczynnik"])
coef["Typ"] = np.where(coef["Zmienna"].str.startswith("num__"), "liczbowa (na 1 odch. std.)", "kategoria (vs. pozostałe)")
coef["Zmienna"] = (coef["Zmienna"].str.replace("num__", "", regex=False)
                   .str.replace("cat__", "", regex=False))
coef = coef.reindex(coef["Współczynnik"].abs().sort_values(ascending=False).index)
header("REGRESJA LOGISTYCZNA")
print(f"Wyraz wolny: {clf.intercept_[0]:+.3f}   |   liczba zmiennych po kodowaniu: {len(names)}   |   C={clf.C}")
print("Interpretacja OR: >1 zwiększa szansę nieprzyjazdu, <1 zmniejsza.")
print(coef.head(25).round(3).to_string(index=False))
coef.to_csv(OUT / "logistic_coefficients.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(9, 7))
top = coef.head(20).iloc[::-1]
ax.barh(top["Zmienna"], top["Współczynnik"],
        color=[COL["critical"] if v > 0 else COL["s1"] for v in top["Współczynnik"]], height=0.65)
ax.axvline(0, color=COL["neutral"], lw=1)
ax.grid(axis="y", visible=False)
titles(ax, "Regresja logistyczna - 20 największych współczynników",
       "Czerwone zwiększają ryzyko nieprzyjazdu, niebieskie zmniejszają (log-odds)")
fig.tight_layout()
show_fig(fig, "logistic_coefficients")

# %% 7. Heurystyka segmentowa (punkt odniesienia "jak dziś")
header("HEURYSTYKA SEGMENTOWA - historyczny odsetek nieprzyjazdów w segmencie")
tab = seg.table_.rename("Prawdopodobieństwo").reset_index().sort_values("Prawdopodobieństwo", ascending=False)
print(f"Liczba segmentów: {len(tab)}, średnia globalna: {seg.global_:.3f}")
print(tab.head(15).round(3).to_string(index=False))

# %% 8. Jedna rezerwacja w trzech modelach + wyjaśnienie LightGBM (SHAP)
example = te[(te["days_to_arrival"] == 14)].sort_values("p_lgbm", ascending=False).iloc[[0, len(te[te["days_to_arrival"] == 14]) // 2]]
contrib = model.contributions(example)
header("PRZYKŁADOWE REZERWACJE (D-14): prognozy wszystkich modeli")
cols = ["booking_id", "hotel", "arrival", "lead_time", "market_segment", "deposit_type", "customer_type",
        "previous_cancellations", "p_segment", "p_logistic", "p_lgbm", "no_arrival"]
print(example[cols].assign(arrival=example["arrival"].dt.strftime("%Y-%m-%d")).round(3).to_string(index=False))
for i, (_, row) in enumerate(example.iterrows()):
    c = contrib.iloc[i].sort_values(key=abs, ascending=False).head(6)
    print(f"\nRezerwacja {row['booking_id']}: 6 największych wkładów cech (SHAP, log-odds; + zwiększa ryzyko):")
    for f, v in c.items():
        print(f"   {FEATURE_LABELS_PL.get(f, f):35s} = {str(model.features(example.iloc[[i]])[f].iloc[0]):12s} {v:+.3f}")

# %% 9. Własna, hipotetyczna rezerwacja - "co jeśli"
new = pd.DataFrame([{
    "hotel": "City Hotel", "lead_time": 120, "arrival": pd.Timestamp("2017-07-14"),
    "arrival_date_week_number": 28, "stays_in_weekend_nights": 2, "stays_in_week_nights": 1,
    "nights": 3, "adults": 2, "children": 0, "babies": 0, "meal": "BB",
    "market_segment": "Online TA", "distribution_channel": "TA/TO", "is_repeated_guest": 0,
    "previous_cancellations": 0, "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
    "deposit_type": "No Deposit", "agent_cat": "A9", "has_company": 0, "days_in_waiting_list": 0,
    "customer_type": "Transient", "adr": 120.0,
}])
header("HIPOTETYCZNA REZERWACJA - ryzyko w kolejnych dniach przed przyjazdem")
rows = []
for h in [120, 90, 60, 30, 14, 7, 3, 1, 0]:
    r = new.assign(days_to_arrival=h)
    rows.append({"Dni do przyjazdu": h,
                 "LightGBM + kalibracja": float(model.predict(r)[0]),
                 "Regresja logistyczna": float(lr.predict_proba(_lr_frame(make_features(r)))[0, 1]),
                 "Heurystyka": float(seg.predict(r)[0])})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nZmień pola w 'new' (np. deposit_type='Non Refund', previous_cancellations=1) i uruchom komórkę ponownie.")

print(f"\nWykresy i tabele zapisano w: {OUT}")
