import numpy as np
import pandas as pd

from hurdle_pipeline import engineer, make_frame


def test_rolling_features_do_not_use_current_rainfall():
    index = pd.date_range("2020-01-01", periods=20)
    frame = pd.DataFrame({
        "Rainfall_mm": np.arange(20, dtype=float),
        "Relative_Humidity_percent": np.full(20, 60.0),
        "Surface_Pressure_kPa": np.full(20, 94.0),
        "Temperature_C": np.full(20, 25.0),
        "Temperature_Max_C": np.full(20, 30.0),
        "Temperature_Min_C": np.full(20, 20.0),
        "Wind_Speed_mps": np.full(20, 2.0),
        "Solar_Radiation": np.full(20, 15.0),
        "Dew_Point_C": np.full(20, 15.0),
    }, index=index)
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("Rainfall_mm")] = 999
    pd.testing.assert_frame_equal(engineer(frame).iloc[:-1], engineer(changed).iloc[:-1])


def test_targets_are_future_dates():
    index = pd.date_range("2020-01-01", periods=20)
    frame = pd.DataFrame({
        "Rainfall_mm": np.arange(20, dtype=float),
        "Relative_Humidity_percent": 60.0, "Surface_Pressure_kPa": 94.0,
        "Temperature_C": 25.0, "Temperature_Max_C": 30.0,
        "Temperature_Min_C": 20.0, "Wind_Speed_mps": 2.0,
        "Solar_Radiation": 15.0, "Dew_Point_C": 15.0,
    }, index=index)
    result = make_frame(frame, 3)
    assert (result["target_date"] > result.index).all()
    assert (result["target_date"] - result.index).dt.days.eq(3).all()
