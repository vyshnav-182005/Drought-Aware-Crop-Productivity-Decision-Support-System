"""
Label every (district, season, year) from 2000-2025 with the contingency-plan
conditions it actually experienced, using the observed daily weather.

This is the graph's source for (District, Season, Year) -EXPERIENCED->
ConditionDetail edges. It is a deliberately simple, transparent rule set that
uses the contingency plans' OWN definitions, so the labels line up with the
graph vocabulary one-to-one. If Stage 4's classifier later writes a file with
the same columns to outputs/condition_classification/season_conditions.csv,
build_graph.py uses that file instead (no graph schema change).

Rules (all thresholds are on daily rainfall, data/weather/clean/*.csv):
  rainy day            >= 2.5 mm  (IMD rainy day; the plans' own "rainless
                                    (>2.5 mm)" threshold)
  monsoon onset        first day from 1 June whose 3-day total is >= 20 mm and
                       which is not followed by a run of >= 14 rainless days (the
                       plans' own dry-spell length) in the next 30 days (searched until 31 Aug; none = no onset)
  onset delay          weeks after 8 June (the plans equate "delay by 2 weeks"
                       with the 4th week of June, i.e. normal onset in the 2nd
                       week of June). Buckets 2/4/6/8 weeks as in the plans.
  normal onset +
  dry spell            onset not delayed (< 2 weeks) and a run of >= 15
                       rainless days starting within 30 days of onset
  mid-season drought   a run of >= 14 rainless days (plans: "consecutive 2
                       weeks rainless") starting 31-60 days after onset
                       (vegetative stage) or 61-90 days (flowering/fruiting)
  terminal drought     15 Sep - 15 Oct rainfall < 50% of the district's own
                       2000-2025 mean for that window ("early withdrawal")
  unusual rains        any 3-day total >= 150 mm (Kharif) or >= 64.5 mm
                       (Rabi, Nov-Feb: unseasonal) -> "continuous high
                       rainfall in a short span"
Season windows: Kharif 1 Jun - 15 Oct, Rabi 16 Oct - 28 Feb (Rabi year = the
sowing year, so Rabi 2015 = Rabi 2015-16). Only seasons fully covered by the
weather file are labelled (Kharif up to the last year with data past 15 Oct,
Rabi up to the last year with data past 28 Feb of the next year).

Also writes season_weather_summary.csv: seasonal rainfall, % of the district
mean, IMD-style rainfall status, onset date, longest dry spell.

Run from the project root:  python3 codes/knowledge_graph/label_seasons.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from district_name_map import CANONICAL_DISTRICTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WEATHER = ROOT / "data" / "weather" / "clean"
OUT = ROOT / "outputs" / "knowledge_graph"
RAINY = 2.5
BASIS = "kg_rule_labeller_v1"


def runs_of_dry(rain: pd.Series):
    """(start_date, length) of every run of consecutive rainless days."""
    dry = (rain < RAINY).to_numpy()
    out, start = [], None
    for i, d in enumerate(dry):
        if d and start is None:
            start = i
        if not d and start is not None:
            out.append((rain.index[start], i - start))
            start = None
    if start is not None:
        out.append((rain.index[start], len(dry) - start))
    return out


def find_onset(rain: pd.Series, year: int):
    r3 = rain.rolling(3).sum().shift(-2)
    for day in pd.date_range(f"{year}-06-01", f"{year}-08-31"):
        if r3.get(day, 0) >= 20:
            nxt = rain[day + pd.Timedelta(days=3): day + pd.Timedelta(days=32)]
            if all(length < 14 for _, length in runs_of_dry(nxt)):
                return day
    return None


def status(pct):
    if pct >= 60: return "Large excess"
    if pct >= 20: return "Excess"
    if pct > -20: return "Normal"
    if pct > -60: return "Deficient"
    return "Large deficient"


def label_district(district):
    df = pd.read_csv(WEATHER / f"{district.lower()}_weather.csv", parse_dates=["DATE"]).set_index("DATE")
    rain = df["Rainfall_mm"].fillna(0)
    years = sorted(y for y in set(rain.index.year) if pd.Timestamp(f"{y}-10-15") <= rain.index.max())

    term_window = lambda y: rain[f"{y}-09-15":f"{y}-10-15"].sum()
    term_mean = np.mean([term_window(y) for y in years])
    monsoon_mean = np.mean([rain[f"{y}-06-01":f"{y}-09-30"].sum() for y in years])
    rabi_years = [y for y in years if pd.Timestamp(f"{y + 1}-02-28") <= rain.index.max()]
    rabi_mean = np.mean([rain[f"{y}-10-16":f"{y + 1}-02-28"].sum() for y in rabi_years])

    labels, summary = [], []

    def add(season, year, cid, **ev):
        labels.append({"district": district, "season": season, "year": year,
                       "condition_detail_id": cid, "basis": BASIS,
                       "evidence": "; ".join(f"{k}={v}" for k, v in ev.items())})

    for y in years:
        k = rain[f"{y}-06-01":f"{y}-10-15"]
        onset = find_onset(rain, y)
        delay = (onset - pd.Timestamp(f"{y}-06-08")).days / 7 if onset is not None else 99
        bucket = next((b for b in (8, 6, 4, 2) if delay >= b), None)
        if bucket:
            add("Kharif", y, f"onset_delay_{bucket}w",
                onset=onset.date() if onset is not None else "none_by_31_Aug", delay_weeks=round(min(delay, 99), 1))
        runs = runs_of_dry(k)
        if onset is not None:
            for start, length in runs:
                after = (start - onset).days
                if not bucket and 0 < after <= 30 and length >= 15:
                    add("Kharif", y, "normal_onset_dry_spell", spell_start=start.date(), spell_days=length)
                elif 31 <= after <= 60 and length >= 14:
                    add("Kharif", y, "mid_dry_spell_vegetative", spell_start=start.date(), spell_days=length)
                elif 61 <= after <= 90 and length >= 14:
                    add("Kharif", y, "mid_dry_spell_flowering", spell_start=start.date(), spell_days=length)
        tw = term_window(y)
        if tw < 0.5 * term_mean:
            add("Kharif", y, "terminal_drought", sep15_oct15_mm=round(tw, 1),
                district_mean_mm=round(term_mean, 1))
        r3 = k.rolling(3).sum()
        if r3.max() >= 150:
            add("Kharif", y, "continuous_high_rainfall", max_3day_mm=round(r3.max(), 1), date=r3.idxmax().date())
        mon = rain[f"{y}-06-01":f"{y}-09-30"].sum()
        pct = (mon / monsoon_mean - 1) * 100
        summary.append({"district": district, "season": "Kharif", "year": y,
                        "rainfall_mm": round(k.sum(), 1), "monsoon_jun_sep_mm": round(mon, 1),
                        "pct_departure_from_mean": round(pct, 1), "rainfall_status": status(pct),
                        "onset_date": onset.date() if onset is not None else "",
                        "onset_delay_weeks": round(delay, 1) if onset is not None else "",
                        "longest_dry_spell_days": max((l for _, l in runs), default=0),
                        "max_3day_rain_mm": round(r3.max(), 1)})

    for y in rabi_years:
        rb = rain[f"{y}-10-16":f"{y + 1}-02-28"]
        r3 = rain[f"{y}-11-01":f"{y + 1}-02-28"].rolling(3).sum()
        if r3.max() >= 64.5:
            add("Rabi", y, "continuous_high_rainfall", max_3day_mm=round(r3.max(), 1),
                date=r3.idxmax().date(), note="unseasonal")
        pct = (rb.sum() / rabi_mean - 1) * 100
        summary.append({"district": district, "season": "Rabi", "year": y,
                        "rainfall_mm": round(rb.sum(), 1), "monsoon_jun_sep_mm": "",
                        "pct_departure_from_mean": round(pct, 1), "rainfall_status": status(pct),
                        "onset_date": "", "onset_delay_weeks": "",
                        "longest_dry_spell_days": max((l for _, l in runs_of_dry(rb)), default=0),
                        "max_3day_rain_mm": round(r3.max(), 1)})
    return labels, summary


def main():
    labels, summary = [], []
    for d in CANONICAL_DISTRICTS:
        a, b = label_district(d)
        labels += a
        summary += b
    pd.DataFrame(labels).to_csv(OUT / "season_conditions.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "season_weather_summary.csv", index=False)
    lab = pd.DataFrame(labels)
    print(f"season_conditions.csv: {len(lab)} labels over {len(summary)} district-seasons")
    print(lab.groupby("condition_detail_id").size().to_string())


if __name__ == "__main__":
    main()
