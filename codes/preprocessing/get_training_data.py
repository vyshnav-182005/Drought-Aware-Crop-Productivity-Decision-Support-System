import requests
import pandas as pd
from datetime import date

# ============================================
# District coordinates - Maharashtra
# ============================================
locations = {
    "Beed": {
        "latitude": 18.9891,
        "longitude": 75.7601
    },
    "Latur": {
        "latitude": 18.4088,
        "longitude": 76.5604
    },
    "Osmanabad": {
        "latitude": 18.1860,
        "longitude": 76.0419
    },
    "Solapur": {
        "latitude": 17.6599,
        "longitude": 75.9064
    },
    "Ahmednagar": {
        "latitude": 19.0948,
        "longitude": 74.7480
    }
}

# ============================================
# Date range
# This is the TRAINING data for your forecasting models (SARIMA/XGBoost/
# LSTM/TFT) -- it is historical, not a forecast itself. The model learns
# patterns from this data; its predictions are the forecast.
# END_DATE is set to run through today automatically, instead of stopping
# at 2025-12-31, so training data is as recent as possible.
# ============================================
START_DATE = "20000101"
END_DATE = date.today().strftime("%Y%m%d")

# ============================================
# Weather parameters
# Rainfall and temperature are the two variables the project actually
# needs (drought/dry-spell detection + crop heat-stress signal). Humidity,
# wind, pressure and radiation are kept too since they are useful predictor
# features (lag/rolling inputs) for forecasting rainfall and temperature,
# even though they are not themselves forecast targets.
# ============================================
parameters = [
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS10M",
    "PS",
    "ALLSKY_SFC_SW_DWN"
]

parameters_str = ",".join(parameters)

# ============================================
# NASA POWER API
# ============================================
url = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ============================================
# Download and save each district separately
# ============================================
for district, coords in locations.items():

    print(f"\nDownloading {district} ({START_DATE} to {END_DATE})...")

    params = {
        "parameters": parameters_str,
        "community": "AG",
        "longitude": coords["longitude"],
        "latitude": coords["latitude"],
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
        "time-standard": "LST"
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    print("Status code:", response.status_code)

    response.raise_for_status()

    data = response.json()

    # ============================================
    # Convert to DataFrame
    # ============================================
    weather = pd.DataFrame(
        data["properties"]["parameter"]
    )

    weather.index.name = "DATE"
    weather.reset_index(inplace=True)

    weather["DATE"] = pd.to_datetime(
        weather["DATE"],
        format="%Y%m%d"
    )

    # NASA POWER uses -999 as a missing-value sentinel; make sure any
    # such rows are flagged as NaN rather than silently used as real
    # readings when you get to feature engineering / training later.
    weather.replace(-999, pd.NA, inplace=True)

    # ============================================
    # Rename columns
    # ============================================
    weather.rename(columns={
        "PRECTOTCORR": "Rainfall_mm",
        "T2M": "Temperature_C",
        "T2M_MAX": "Temperature_Max_C",
        "T2M_MIN": "Temperature_Min_C",
        "RH2M": "Relative_Humidity_percent",
        "WS10M": "Wind_Speed_mps",
        "PS": "Surface_Pressure_kPa",
        "ALLSKY_SFC_SW_DWN": "Solar_Radiation"
    }, inplace=True)

    # ============================================
    # Save separate CSV
    # ============================================
    filename = f"{district.lower()}_weather_training_data.csv"

    weather.to_csv(
        filename,
        index=False
    )

    print(f"Saved: {filename}")
    print(f"Rows: {len(weather)}")
    print(f"Columns: {len(weather.columns)}")
    print(f"Missing values: {int(weather.isna().sum().sum())}")

print("\nAll 5 district training datasets downloaded successfully!")
print("Next step (when you're ready): build lag/rolling features from")
print("these files as input to SARIMA/XGBoost/LSTM/TFT.")
