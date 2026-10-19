import requests
import pandas as pd

# ============================================
# District coordinates - Maharashtra
# (same points as your historical NASA POWER script)
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
# Forecast window
# 16 is the hard maximum this API allows. See the note printed at the end
# of this script for why you should treat only the first ~7 days as usable.
# There is no "6 months" option here or anywhere else at daily resolution --
# not a setting to raise, a hard ceiling of forecast science.
# ============================================
FORECAST_DAYS = 16

# ============================================
# Weather parameters
# Chosen to mirror your historical NASA POWER columns as closely as this
# forecast API allows, so the two files can feed the same downstream code.
# Requested hourly, then aggregated to daily below (Open-Meteo's forecast
# endpoint does not expose a "daily surface pressure / solar radiation"
# field directly, only hourly).
# ============================================
hourly_parameters = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "windspeed_10m",
    "shortwave_radiation"
]

parameters_str = ",".join(hourly_parameters)

# ============================================
# Open-Meteo Forecast API (live NWP forecast - free, no key)
# ============================================
url = "https://api.open-meteo.com/v1/forecast"

# ============================================
# Download and save each district separately
# ============================================
for district, coords in locations.items():

    print(f"\nFetching forecast for {district}...")

    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": parameters_str,
        "forecast_days": FORECAST_DAYS,
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    print("Status code:", response.status_code)

    response.raise_for_status()

    data = response.json()["hourly"]

    # ============================================
    # Convert to DataFrame (hourly rows)
    # ============================================
    hourly = pd.DataFrame(data)
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["DATE"] = hourly["time"].dt.date

    # ============================================
    # Aggregate hourly -> daily, matching the grain of your historical file
    # ============================================
    weather = hourly.groupby("DATE").agg(
        Temperature_C=("temperature_2m", "mean"),
        Temperature_Max_C=("temperature_2m", "max"),
        Temperature_Min_C=("temperature_2m", "min"),
        Rainfall_mm=("precipitation", "sum"),
        Relative_Humidity_percent=("relative_humidity_2m", "mean"),
        Wind_Speed_mps=("windspeed_10m", "mean"),
        Surface_Pressure_kPa=("surface_pressure", "mean"),
        # W/m^2 hourly mean -> MJ/m^2/day, to match ALLSKY_SFC_SW_DWN's units
        Solar_Radiation=("shortwave_radiation", lambda s: s.mean() * 0.0036 * 24)
    ).reset_index()

    # ============================================
    # Save separate CSV
    # ============================================
    filename = f"{district.lower()}_weather_forecast.csv"

    weather.to_csv(
        filename,
        index=False
    )

    print(f"Saved: {filename}")
    print(f"Rows: {len(weather)}")
    print(f"Columns: {len(weather.columns)}")

print("\nAll 5 district forecasts downloaded successfully!")
print("\nNOTE: days 1-7 of this data are reliable enough to use in the")
print("project; days 8-16 are included by the API but accuracy drops fast")
print("and should not be relied on for a season-level claim.")
