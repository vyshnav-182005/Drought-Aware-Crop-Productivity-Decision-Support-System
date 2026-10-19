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
# END_DATE now tracks today automatically instead of a fixed 2025-12-31,
# so re-running this script later always pulls up through the present.
# ============================================
START_DATE = "20000101"
END_DATE = date.today().strftime("%Y%m%d")

# ============================================
# Weather parameters
# Original 8 left untouched. Three added at the end for forecasting:
#   - T2MDEW    : dew point @ 2m -- a direct moisture measure, a stronger
#                 precipitation predictor than relative humidity alone
#   - CLOUD_AMT : cloud amount (%) -- directly informative for near-term
#                 rain likelihood and next-day solar radiation
#   - WS50M     : wind speed @ 50m -- sits above surface friction, so it
#                 carries a bit more of the larger-scale flow signal than
#                 WS10M alone (still no substitute for real upper-air data,
#                 but the best proxy POWER offers)
# ============================================
parameters = [
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS10M",
    "PS",
    "ALLSKY_SFC_SW_DWN",
    "T2MDEW",
    "CLOUD_AMT",
    "WS50M"
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

    print(f"\nDownloading {district}...")

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

    # NASA POWER uses -999 as a missing-value sentinel -- flag it as NaN
    # so it isn't silently used as a real reading (matters more now that
    # END_DATE reaches right up to the present, where a day or two of
    # data can lag behind processing).
    weather.replace(-999, pd.NA, inplace=True)

    # ============================================
    # Rename columns
    # Original 8 renames untouched; 3 new ones added for the new params.
    # ============================================
    weather.rename(columns={
        "PRECTOTCORR": "Rainfall_mm",
        "T2M": "Temperature_C",
        "T2M_MAX": "Temperature_Max_C",
        "T2M_MIN": "Temperature_Min_C",
        "RH2M": "Relative_Humidity_percent",
        "WS10M": "Wind_Speed_mps",
        "PS": "Surface_Pressure_kPa",
        "ALLSKY_SFC_SW_DWN": "Solar_Radiation",
        "T2MDEW": "Dew_Point_C",
        "CLOUD_AMT": "Cloud_Amount_percent",
        "WS50M": "Wind_Speed_50m_mps"
    }, inplace=True)

    # ============================================
    # Save separate CSV
    # ============================================
    filename = f"{district.lower()}_weather_2000_2025.csv"

    weather.to_csv(
        filename,
        index=False
    )

    print(f"Saved: {filename}")
    print(f"Rows: {len(weather)}")
    print(f"Columns: {len(weather.columns)}")

print("\nAll 5 district datasets downloaded successfully!")
