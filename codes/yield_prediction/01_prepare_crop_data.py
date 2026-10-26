"""
Stage 1: Crop Data Ingestion and Harmonization
Tasks 1, 2, 3:
- Load DES crop HTML-mislabeled .xls file with pandas.read_html
- Standardize district names (Dharashiv -> Osmanabad, Ahilyanagar -> Ahmednagar, Bid -> Beed)
- Standardize crop names (Bajra = Pearl millet)
- Reshape into canonical long table: [district, crop, season, year, area, production, yield]
- Retain only Kharif and Rabi
- Handle missing cells as NaN and drop
- Support merging Rabi Gram / Chickpea
Output: data/crop_production/cleaned_crop_yield.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from common import CROP_DIR as DATA_DIR, CLEANED_CROP_FILE as OUTPUT_FILE

# District canonicalization map
DISTRICT_MAP = {
    "1. Ahilyanagar": "Ahmednagar",
    "Ahilyanagar": "Ahmednagar",
    "Ahmednagar": "Ahmednagar",
    "2. Beed": "Beed",
    "Bid": "Beed",
    "Beed": "Beed",
    "3. Dharashiv": "Osmanabad",
    "Dharashiv": "Osmanabad",
    "Osmanabad": "Osmanabad",
    "4. Latur": "Latur",
    "Latur": "Latur",
    "5. Solapur": "Solapur",
    "Solapur": "Solapur"
}

# Crop canonicalization map
CROP_MAP = {
    "Arhar/Tur": "Arhar/Tur",
    "Bajra": "Pearl millet",
    "Pearl millet": "Pearl millet",
    "Jowar": "Jowar",
    "Soyabean": "Soyabean",
    "Sunflower": "Sunflower",
    "Gram": "Gram",
    "Chickpea": "Gram"
}


def load_and_clean_des_file(des_path: Path) -> pd.DataFrame:
    """Reads HTML-based .xls DES file and reshapes to long table."""
    print(f"Reading DES crop file from: {des_path}")
    dfs = pd.read_html(str(des_path))
    if not dfs:
        raise ValueError(f"No HTML table found in {des_path}")
    df = dfs[0]

    raw_district = df.iloc[:, 1].values
    raw_year = df.iloc[:, 2].values

    # Extract all crop columns present
    # Looking for multi-index tuple columns (crop, season, metric)
    crops_in_table = [c for c in df.columns.levels[0] if c not in ["State", "District", "Year"]]

    records = []
    for idx in range(len(df)):
        d_val = str(raw_district[idx]).strip()
        y_val = str(raw_year[idx]).strip()

        district = DISTRICT_MAP.get(d_val, d_val)
        # Parse start year from '2000 - 2001' or '2000-01'
        year = int(y_val.split("-")[0].strip())

        for crop in crops_in_table:
            std_crop = CROP_MAP.get(crop, crop)
            for season in ["Kharif", "Rabi"]:
                area_col = (crop, season, "Area (Hectare)")
                prod_col = (crop, season, "Production (Tonnes)")
                yield_col = (crop, season, "Yield (Tonne/Hectare)")

                if area_col in df.columns:
                    area = df.loc[idx, area_col]
                    prod = df.loc[idx, prod_col] if prod_col in df.columns else np.nan
                    yld = df.loc[idx, yield_col] if yield_col in df.columns else np.nan

                    # Clean string representations of numbers/missing
                    def clean_val(v):
                        if pd.isna(v):
                            return np.nan
                        if isinstance(v, str):
                            v = v.replace(",", "").strip()
                            if v in ["-", "", "NR", "NA", "NaN"]:
                                return np.nan
                        try:
                            val = float(v)
                            # Missing values in DES are sometimes recorded as 0 or negative;
                            # Area or production cannot be <= 0 for active recorded farming
                            return val if val > 0 else np.nan
                        except ValueError:
                            return np.nan

                    area_clean = clean_val(area)
                    prod_clean = clean_val(prod)

                    # Only record valid rows where area and production exist
                    if pd.notna(area_clean) and pd.notna(prod_clean):
                        calculated_yield = prod_clean / area_clean
                        records.append({
                            "district": district,
                            "crop": std_crop,
                            "season": season,
                            "year": year,
                            "area": area_clean,
                            "production": prod_clean,
                            "yield": calculated_yield
                        })

    long_df = pd.DataFrame(records)
    print(f"Extracted {len(long_df)} valid district-crop-season-year records.")
    return long_df


def add_gram_data_if_available(yield_df: pd.DataFrame) -> pd.DataFrame:
    """Checks for or loads Rabi Gram/Chickpea data for the 5 districts."""
    # Check for downloaded DES Gram report or rabi_gram.csv
    gram_xls = DATA_DIR / "horizontal_crop_vertical_year_report (1).xls"
    gram_csv = DATA_DIR / "rabi_gram.csv"

    if gram_xls.exists():
        print(f"Found downloaded DES Gram file: {gram_xls.name}")
        gdf = load_and_clean_des_file(gram_xls)
        yield_df = pd.concat([yield_df, gdf], ignore_index=True)
        print(f"Successfully integrated {len(gdf)} Rabi Gram records into the yield table!")
    elif gram_csv.exists():
        print(f"Loading Gram data from {gram_csv}...")
        gdf = pd.read_csv(gram_csv)
        gdf["district"] = gdf["district"].map(lambda d: DISTRICT_MAP.get(d, d))
        gdf["crop"] = "Gram"
        gdf["season"] = "Rabi"
        if "yield" not in gdf.columns and "production" in gdf.columns and "area" in gdf.columns:
            gdf["yield"] = gdf["production"] / gdf["area"]
        yield_df = pd.concat([yield_df, gdf[["district", "crop", "season", "year", "area", "production", "yield"]]], ignore_index=True)
    else:
        print("Note: Optional rabi_gram file not found on disk. Proceeding with existing 7 crop-season series from DES.")
    return yield_df


def main():
    des_xls = DATA_DIR / "horizontal_crop_vertical_year_report.xls"
    if not des_xls.exists():
        print(f"Error: {des_xls} not found.")
        sys.exit(1)

    df_cleaned = load_and_clean_des_file(des_xls)
    df_cleaned = add_gram_data_if_available(df_cleaned)

    # Sort deterministically
    df_cleaned = df_cleaned.sort_values(by=["district", "crop", "season", "year"]).reset_index(drop=True)

    # Save cleaned file
    df_cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved standardized crop yield table to: {OUTPUT_FILE}")
    print("\nSummary by Crop & Season:")
    print(df_cleaned.groupby(["crop", "season"]).agg(
        records=("year", "count"),
        min_year=("year", "min"),
        max_year=("year", "max"),
        mean_yield=("yield", "mean"),
        mean_area=("area", "mean")
    ))


if __name__ == "__main__":
    main()
