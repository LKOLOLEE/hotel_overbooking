"""
Konfiguracja projektu: ścieżki, parametry modelu, parametry biznesowe
oraz WSZYSTKIE założenia dla danych syntetycznych.

Każdy parametr oznaczony [SYNTETYCZNE] nie pochodzi z publicznego zbioru
'Hotel Booking Demand' - jest założeniem, które w prawdziwym wdrożeniu należy
zastąpić danymi hotelu (PMS, finanse, revenue management).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Ścieżki
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "hotel_bookings.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
REPORTS = OUTPUTS / "reports"
DAILY = OUTPUTS / "daily"
SITE = OUTPUTS / "site"            # statyczna strona do publikacji online (index.html + archiwum)
LOGS = ROOT / "logs"

# Źródło danych publicznych (Antonio, Almeida, Nunes, 2019, Data in Brief,
# licencja CC BY 4.0; wersja oczyszczona z repozytorium TidyTuesday).
DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2020/2020-02-11/hotels.csv"
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Podział czasowy (po dacie przyjazdu) - NIGDY podział losowy
# ---------------------------------------------------------------------------
TRAIN_END = "2016-06-30"   # trening: przyjazdy 2015-07-01 .. 2016-06-30
VALID_END = "2016-12-31"   # walidacja + kalibracja: 2016-07-01 .. 2016-12-31
# test (backtest): przyjazdy 2017-01-01 .. 2017-08-31

# Migawki: ile dni przed przyjazdem "robimy zdjęcie" stanu rezerwacji.
SNAPSHOT_HORIZONS = [0, 1, 3, 7, 14, 30, 60, 90, 120, 180]

# Cechy, których NIE wolno używać (wyciek informacji z przyszłości).
# - country: w tym zbiorze kraj jest uzupełniany przy zameldowaniu, więc
#   rezerwacje anulowane mają nienaturalnie często 'PRT' (znany artefakt).
# - booking_changes, assigned_room_type, required_car_parking_spaces,
#   total_of_special_requests: wartości końcowe, znane dopiero po fakcie
#   lub aktualizowane w trakcie życia rezerwacji.
LEAKY_COLUMNS = [
    "country", "booking_changes", "assigned_room_type",
    "required_car_parking_spaces", "total_of_special_requests",
    "reservation_status", "reservation_status_date", "is_canceled",
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
LGBM_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=300,
    cat_smooth=50,
    min_data_per_group=200,
    cat_l2=10,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=RANDOM_STATE,
)
LGBM_NUM_ROUNDS = 2000
LGBM_EARLY_STOPPING = 100
TOP_AGENTS = 40          # agenci spoza top-N -> 'other'

# ---------------------------------------------------------------------------
# Pojemność hotelu
# ---------------------------------------------------------------------------
# Zbiór nie zawiera liczby pokoi. None = oszacuj jako maksymalne
# zrealizowane obłożenie (dane wyraźnie "dobijają" do stałego sufitu).
# W prawdziwym wdrożeniu: wpisać liczbę pokoi z PMS, np. {"City Hotel": 230}.
CAPACITY_OVERRIDE = None

# Okno nocy do analiz (pierwsze dni lipca 2015 nie mają pełnej historii
# pobytów rozpoczętych przed początkiem zbioru).
NIGHTS_START = "2015-07-15"
NIGHTS_END = "2017-08-31"

# ---------------------------------------------------------------------------
# Overbooking - parametry biznesowe  [SYNTETYCZNE]
# ---------------------------------------------------------------------------
DECISION_HORIZON_DAYS = 3        # decyzja o dodatkowej sprzedaży na D-3
MAX_EXTRA_BOOKINGS = 40          # maksymalna liczba dodatkowych rezerwacji / noc

# Koszt relokacji gościa ("walk"): pokój w innym hotelu, transfer,
# rekompensata, utrata lojalności. Wyrażony względem ADR danej nocy.
WALK_COST_ADR_MULTIPLIER = 2.0   # [SYNTETYCZNE]
WALK_COST_FIXED_EUR = 150.0      # [SYNTETYCZNE]

# Średnie wydatki dodatkowe gościa na pokój-noc (F&B, bar, SPA itp.)
ANCILLARY_PER_ROOM_NIGHT = {     # [SYNTETYCZNE]
    "Resort Hotel": 35.0,
    "City Hotel": 20.0,
}

# Niezaspokojony popyt (odrzucone zapytania) w noce wyprzedane.
# Zbiór zawiera tylko przyjęte rezerwacje, więc popyt odrzucony
# generujemy syntetycznie: Poisson(lambda * pojemność) w noce,
# w których zrealizowane obłożenie >= SOLD_OUT_THRESHOLD * pojemność.
SOLD_OUT_THRESHOLD = 0.95        # [SYNTETYCZNE]
EXTRA_DEMAND_LAMBDA = 0.10       # [SYNTETYCZNE] 10% pojemności

# Polityka porównawcza: stały overbooking (jak w wielu hotelach)
STATIC_OVERBOOKING_PCT = 0.05

# Okno dla prognozy "pickup" (rezerwacje, które dopiero wpłyną)
PICKUP_LOOKBACK_WEEKS = 8

# ---------------------------------------------------------------------------
# Działania prewencyjne (outreach) - eksperyment A/B  [SYNTETYCZNE]
# ---------------------------------------------------------------------------
OUTREACH_HORIZON_DAYS = 14       # kontakt z gościem na D-14
OUTREACH_TREATMENT_SHARE = 0.5   # połowa grupa testowa, połowa kontrolna
OUTREACH_MAX_EFFECT = 0.30       # [SYNTETYCZNE] maks. względny spadek anulacji
OUTREACH_COST_PER_CONTACT = 1.0  # [SYNTETYCZNE] EUR (automatyczny e-mail + obsługa)
OUTREACH_DISCOUNT_PCT = 0.05     # [SYNTETYCZNE] rabat za przedpłatę
OUTREACH_DISCOUNT_TAKE_UP = 0.15 # [SYNTETYCZNE] odsetek przyjmujących rabat
# Prawdopodobieństwo, że zwolniony pokój i tak zostałby odsprzedany
RESALE_PROBABILITY = 0.50        # [SYNTETYCZNE]

# ---------------------------------------------------------------------------
# Polityki anulacji / depozytów - symulacja "what-if"  [SYNTETYCZNE]
# ---------------------------------------------------------------------------
POLICY_CANCEL_REDUCTION = 0.40   # wymóg przedpłaty zmniejsza anulacje o 40%
POLICY_BOOKING_LOSS = 0.08       # ... ale zniechęca 8% rezerwujących
POLICY_EARLY_RESALE = 0.80       # [SYNTETYCZNE] wcześnie zwolniony pokój zwykle się sprzedaje
POLICY_REPLACEMENT = 0.50        # [SYNTETYCZNE] utraconą rezerwację częściowo zastąpi inny gość
POLICY_MIN_BOOKINGS = 500        # min. liczba rezerwacji w segmencie

# ---------------------------------------------------------------------------
# Business case  [SYNTETYCZNE]
# ---------------------------------------------------------------------------
PROJECT_COST_EUR = 30000.0       # 1 osoba x ~3 miesiące
RUN_COST_EUR_PER_YEAR = 6000.0   # utrzymanie, monitoring, retrening

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
PSI_WARN = 0.10
PSI_ALERT = 0.25

# ---------------------------------------------------------------------------
# Dzień "dzisiejszy" dla demonstracji codziennego uruchomienia
# (musi leżeć w okresie testowym zbioru)
# ---------------------------------------------------------------------------
DEMO_AS_OF = "2017-06-15"
FORWARD_DAYS = 60

# Tryb demonstracyjny "replay" dla automatycznego uruchamiania: dane publiczne kończą się
# 31.08.2017, więc każdy prawdziwy dzień jest mapowany na kolejny dzień okresu testowego.
# Okno kończy się 60 dni przed końcem danych, aby plan na 60 nocy był kompletny.
REPLAY_START = "2017-01-15"
REPLAY_END = "2017-07-02"
REPLAY_ANCHOR = "2026-01-01"       # prawdziwa data odpowiadająca REPLAY_START
ARCHIVE_DAYS = 30                  # ile ostatnich raportów pokazywać w archiwum strony

# ---------------------------------------------------------------------------
# Paleta (zwalidowana pod kątem daltonizmu)
# ---------------------------------------------------------------------------
COLORS = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text2": "#52514e",
    "grid": "#e4e3df",
    "s1": "#2a78d6",   # niebieski
    "s2": "#eb6834",   # pomarańczowy
    "s3": "#1baf7a",   # morski
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
    "neutral": "#b9b8b3",
    "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
}
HOTEL_COLORS = {"City Hotel": COLORS["s1"], "Resort Hotel": COLORS["s2"]}
