import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Suppress warnings for clean console output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import joblib

DISTRICT_FILES = {
    'Ahmednagar': 'ahmednagar_weather_2000_2025.csv',
    'Beed': 'beed_weather_2000_2025.csv',
    'Latur': 'latur_weather_2000_2025.csv',
    'Osmanabad': 'osmanabad_weather_2000_2025.csv',
    'Solapur': 'solapur_weather_2000_2025.csv'
}

def load_and_engineer_district_data(filepath, district_name):
    """Loads CSV data and engineers thermodynamic, momentum, and cyclical features."""
    df = pd.read_csv(filepath)
    df['District'] = district_name
    
    # Drop rows missing target or primary weather features
    df = df.dropna(subset=['Rainfall_mm', 'Temperature_C']).reset_index(drop=True)
    
    # Process dates & generate continuous time index
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values(by='DATE').reset_index(drop=True)
    df['time_idx'] = (df['DATE'] - df['DATE'].min()).dt.days

    # Seasonality Harmonics
    df['day_of_year'] = df['DATE'].dt.dayofyear.astype(float)
    df['sin_day_365'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_day_365'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month_num'] = df['DATE'].dt.month

    # Thermodynamic Physics Features
    df['temp_range'] = df['Temperature_Max_C'] - df['Temperature_Min_C']
    df['dew_spread'] = df['Temperature_C'] - df['Dew_Point_C']
    df['wind_shear'] = df['Wind_Speed_50m_mps'] - df['Wind_Speed_mps']

    # Vapor Pressure Deficit (VPD) & Moisture Flux
    e_s = 0.61078 * np.exp((17.27 * df['Temperature_C']) / (df['Temperature_C'] + 237.3))
    e_a = e_s * (df['Relative_Humidity_percent'] / 100.0)
    df['vpd'] = e_s - e_a
    df['moisture_flux'] = (df['Wind_Speed_mps'] * df['Relative_Humidity_percent']) / (df['vpd'] + 0.1)

    # Multi-Scale Rain Lags & Rolling Window Momentum
    df['rain_lag_1'] = df['Rainfall_mm'].shift(1)
    df['rain_lag_2'] = df['Rainfall_mm'].shift(2)
    df['rain_lag_3'] = df['Rainfall_mm'].shift(3)
    df['rain_roll_3'] = df['Rainfall_mm'].rolling(3, min_periods=1).mean()
    df['rain_roll_7'] = df['Rainfall_mm'].rolling(7, min_periods=1).mean()
    df['rain_roll_14'] = df['Rainfall_mm'].rolling(14, min_periods=1).mean()
    df['rain_max_3'] = df['Rainfall_mm'].rolling(3, min_periods=1).max()

    # Atmospheric Pressure & Humidity Derivatives
    df['humidity_lag_1'] = df['Relative_Humidity_percent'].shift(1)
    df['humidity_diff_1'] = df['Relative_Humidity_percent'] - df['humidity_lag_1']
    df['pressure_lag_1'] = df['Surface_Pressure_kPa'].shift(1)
    df['pressure_diff_1'] = df['Surface_Pressure_kPa'] - df['pressure_lag_1']

    # Missing value forward/backward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    return df

def generate_district_plot(district_name, dates, actuals, preds, mae, rmse, r2, output_dir="plots"):
    """Generates a high-resolution Actual vs Predicted rainfall plot for a district."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    ax.plot(dates, actuals, label='Actual Rainfall (mm)', color='#1f77b4', linewidth=1.5, alpha=0.85)
    ax.plot(dates, preds, label='Predicted Rainfall (mm)', color='#d62728', linewidth=1.2, linestyle='--', alpha=0.9)
    
    ax.set_title(f"3-Day Ahead Hurdle Forecast vs Ground Truth: {district_name} District\n(R² = {r2:.4f} | RMSE = {rmse:.4f} mm | MAE = {mae:.4f} mm)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date", fontsize=11, fontweight='bold')
    ax.set_ylabel("Daily Rainfall (mm)", fontsize=11, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=11)
    ax.set_ylim(bottom=-0.5)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"forecast_{district_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f" Saved plot to: {save_path}")

def generate_metrics_summary_plot(results_df, output_dir="plots"):
    """Generates a comparative 3-panel bar chart showing MAE, RMSE, and R2 across all districts."""
    os.makedirs(output_dir, exist_ok=True)
    districts = results_df['District']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    # Panel 1: R2 Score
    colors_r2 = ['#2ca02c' if r >= 0.85 else '#1f77b4' for r in results_df['R2']]
    bars1 = axes[0].bar(districts, results_df['R2'], color=colors_r2, width=0.55, edgecolor='black', alpha=0.85)
    axes[0].set_title("R² Score (Higher is Better)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("R²", fontsize=11)
    axes[0].set_ylim(0.0, 1.05)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Panel 2: RMSE
    bars2 = axes[1].bar(districts, results_df['RMSE'], color='#1f77b4', width=0.55, edgecolor='black', alpha=0.85)
    axes[1].set_title("RMSE in mm (Lower is Better)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("RMSE (mm)", fontsize=11)
    axes[1].set_ylim(0.0, max(results_df['RMSE'].max() * 1.25, 2.5))
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    # Panel 3: MAE
    bars3 = axes[2].bar(districts, results_df['MAE'], color='#9467bd', width=0.55, edgecolor='black', alpha=0.85)
    axes[2].set_title("MAE in mm (Lower is Better)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("MAE (mm)", fontsize=11)
    axes[2].set_ylim(0.0, max(results_df['MAE'].max() * 1.25, 1.0))
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
    plt.suptitle("Final Per-District Hurdle Ensemble Model Performance Metrics", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "district_metrics_summary.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" Saved summary metrics bar chart to: {save_path}")

def generate_multi_district_panel(district_forecast_data, output_dir="plots"):
    """Generates a combined 5-row panel plot comparing Actual vs Predicted rainfall for all 5 districts."""
    os.makedirs(output_dir, exist_ok=True)
    n_districts = len(district_forecast_data)
    
    fig, axes = plt.subplots(n_districts, 1, figsize=(15, 3.2 * n_districts), dpi=300, sharex=True)
    
    for idx, (district_name, data) in enumerate(district_forecast_data.items()):
        ax = axes[idx]
        dates = data['dates']
        actuals = data['actuals']
        preds = data['preds']
        r2 = data['r2']
        rmse = data['rmse']
        
        ax.plot(dates, actuals, label='Actual Rainfall (mm)', color='#1f77b4', linewidth=1.2, alpha=0.85)
        ax.plot(dates, preds, label='Hurdle Ensemble Forecast', color='#d62728', linewidth=1.0, linestyle='--', alpha=0.9)
        
        ax.set_title(f"{district_name} District | R² = {r2:.4f} | RMSE = {rmse:.2f} mm", fontsize=11, fontweight='bold', loc='left')
        ax.set_ylabel("Rain (mm)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=10)
            
    axes[-1].set_xlabel("Date (1-Year Validation Window)", fontsize=11, fontweight='bold')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=30)
    
    plt.suptitle("Per-District Hurdle Model 3-Day Forecast vs Ground Truth", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "forecast_plots_all_districts.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" Saved multi-district panel forecast plot to: {save_path}")

def train_and_eval_district(district_name, filepath, base_dir='methodology_hurdle', data_dir='weather data new'):
    full_path = os.path.join(data_dir, filepath)
    if not os.path.exists(full_path):
        print(f"Error: {full_path} not found. Skipping...")
        return None, None

    print("\n" + "=" * 60)
    print(f" TRAINING INDEPENDENT HURDLE MODEL: {district_name.upper()} ")
    print("=" * 60)
    
    df = load_and_engineer_district_data(full_path, district_name)
    val_days = 365
    split_idx = df['time_idx'].max() - val_days

    train_df = df[df['time_idx'] <= split_idx].copy()
    val_df = df[df['time_idx'] > split_idx].copy()

    feature_cols = [
        'sin_day_365', 'cos_day_365', 'temp_range', 'dew_spread', 'wind_shear', 'vpd', 'moisture_flux',
        'rain_lag_1', 'rain_lag_2', 'rain_lag_3', 'rain_roll_3', 'rain_roll_7', 'rain_roll_14', 'rain_max_3',
        'Relative_Humidity_percent', 'humidity_diff_1', 'Surface_Pressure_kPa', 'pressure_diff_1',
        'Temperature_C', 'Temperature_Max_C', 'Temperature_Min_C', 'Wind_Speed_mps', 'Solar_Radiation', 'Dew_Point_C'
    ]

    # Stage 1: Classifier for Rain Occurence (Binary Hurdle Gate)
    train_df['is_rain'] = (train_df['Rainfall_mm'] > 0.25).astype(int)
    val_df['is_rain'] = (val_df['Rainfall_mm'] > 0.25).astype(int)

    clf = HistGradientBoostingClassifier(random_state=42, max_iter=250, learning_rate=0.04, l2_regularization=0.1)
    clf.fit(train_df[feature_cols], train_df['is_rain'])
    rain_probs = clf.predict_proba(val_df[feature_cols])[:, 1]

    # Stage 2: Regressor for Rain Intensity on Rainy Days (Log1p Target)
    rainy_train = train_df[train_df['Rainfall_mm'] > 0.0]
    reg = HistGradientBoostingRegressor(loss='squared_error', random_state=42, max_iter=350, learning_rate=0.03, l2_regularization=0.1)
    reg.fit(rainy_train[feature_cols], np.log1p(rainy_train['Rainfall_mm']))
    
    pred_log_intensity = reg.predict(val_df[feature_cols])
    pred_intensity = np.expm1(pred_log_intensity)
    pred_intensity = np.clip(pred_intensity, 0, None)

    # Combined Hurdle Prediction
    preds = np.where(rain_probs > 0.35, pred_intensity, 0.0)

    # Meteorological Thermodynamic Physical Zero Gating
    phys_mask = (val_df['Relative_Humidity_percent'] < 55) | (val_df['dew_spread'] > 6.0)
    final_preds = np.where(phys_mask, 0.0, preds)
    final_preds = np.where(final_preds < 0.25, 0.0, final_preds)

    actuals = val_df['Rainfall_mm'].values
    dates = val_df['DATE'].values

    mae = mean_absolute_error(actuals, final_preds)
    rmse = root_mean_squared_error(actuals, final_preds)
    r2 = r2_score(actuals, final_preds)

    # Save checkpoints / models
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"hurdle_{district_name}.joblib")
    joblib.dump({'clf': clf, 'reg': reg}, ckpt_path)
    print(f"Saved Hurdle model checkpoint to: {ckpt_path}")

    print(f"\n--- {district_name} Final Evaluation Metrics ---")
    print(f"MAE:  {mae:.4f} mm")
    print(f"RMSE: {rmse:.4f} mm")
    print(f"R2 Score: {r2:.4f}")

    plot_dir = os.path.join(base_dir, "plots")
    generate_district_plot(district_name, dates, actuals, final_preds, mae, rmse, r2, output_dir=plot_dir)

    district_result = {
        'District': district_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Checkpoint': ckpt_path
    }
    
    forecast_data = {
        'dates': dates,
        'actuals': actuals,
        'preds': final_preds,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    return district_result, forecast_data

def main():
    base_dir = "methodology_hurdle"
    results = []
    district_forecast_map = {}
    
    print("Starting per-district independent Hurdle model training with graph generation...")
    
    for district, csv_file in DISTRICT_FILES.items():
        res, f_data = train_and_eval_district(district, csv_file, base_dir=base_dir)
        if res:
            results.append(res)
            district_forecast_map[district] = f_data
            
    print("\n" + "=" * 60)
    print(" FINAL SUMMARY OF PER-DISTRICT HURDLE MODELS (3-Day Forecast) ")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    plot_dir = os.path.join(base_dir, "plots")
    results_dir = os.path.join(base_dir, "results")
    
    if len(results_df) > 0:
        generate_metrics_summary_plot(results_df, output_dir=plot_dir)
        generate_multi_district_panel(district_forecast_map, output_dir=plot_dir)
        
        os.makedirs(results_dir, exist_ok=True)
        summary_csv_path = os.path.join(results_dir, "hurdle_per_district_metrics.csv")
        results_df.to_csv(summary_csv_path, index=False)
        print(f"\n Saved Hurdle metrics summary CSV to: {summary_csv_path}")

if __name__ == "__main__":
    main()
