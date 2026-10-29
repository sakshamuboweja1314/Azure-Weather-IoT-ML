
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Output directory:", output_dir, flush=True)
    print("Generating synthetic IoT weather data...", flush=True)
    rng = np.random.default_rng(42)
    n = 5000
    hour = rng.integers(0, 24, n)
    temperature = 25 + 6 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 2, n)
    humidity = np.clip(65 - 0.8 * (temperature - 25) + rng.normal(0, 12, n), 20, 100)
    pressure = np.clip(1013 - 0.7 * (humidity - 60) + rng.normal(0, 5, n), 980, 1035)
    wind_speed = np.clip(rng.gamma(2, 5, n), 0, 60)
    rainfall = np.where(rng.random(n) < humidity / 180, rng.gamma(1.5, 2, n), 0)
    rainfall = np.clip(rainfall, 0, 30)

    # Simulated target: rain likely/unlikely, generated from programmed rules.
    rain_score = (-3.0 + 0.055 * (humidity - 60) + 0.08 * (1010 - pressure)
                  + 0.06 * wind_speed + 0.30 * np.minimum(rainfall, 5))
    rain_probability = 1 / (1 + np.exp(-rain_score))
    rain_next = (rng.random(n) < rain_probability).astype(int)
    temperature_next = (temperature + 0.7 * np.sin(2 * np.pi * (hour - 12) / 24)
                        - 0.025 * (humidity - 60) + rng.normal(0, 1, n))

    features = ["temperature", "humidity", "pressure", "wind_speed", "rainfall", "hour"]
    df = pd.DataFrame({
        "temperature": temperature, "humidity": humidity, "pressure": pressure,
        "wind_speed": wind_speed, "rainfall": rainfall, "hour": hour,
        "rain_next": rain_next, "temperature_next": temperature_next,
    })
    df.to_csv(output_dir / "weather_data.csv", index=False)

    X = df[features]
    y_rain = df["rain_next"]
    y_temp = df["temperature_next"]
    X_train, X_test, y_rain_train, y_rain_test = train_test_split(
        X, y_rain, test_size=0.2, random_state=42, stratify=y_rain
    )
    y_temp_train = y_temp.loc[X_train.index]
    y_temp_test = y_temp.loc[X_test.index]

    rain_model = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=2,
        random_state=42, class_weight="balanced"
    )
    rain_model.fit(X_train, y_rain_train)
    rain_predictions = rain_model.predict(X_test)
    rain_accuracy = accuracy_score(y_rain_test, rain_predictions)

    temperature_model = RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    temperature_model.fit(X_train, y_temp_train)
    temp_predictions = temperature_model.predict(X_test)

    metrics = {
        "rain_accuracy": float(rain_accuracy),
        "temperature_mae": float(mean_absolute_error(y_temp_test, temp_predictions)),
        "temperature_rmse": float(mean_squared_error(y_temp_test, temp_predictions) ** 0.5),
        "temperature_r2": float(r2_score(y_temp_test, temp_predictions)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    # Save both individual PKL files and a combined bundle.
    joblib.dump(rain_model, output_dir / "rain_model.pkl")
    joblib.dump(temperature_model, output_dir / "temperature_model.pkl")
    joblib.dump({
        "rain_model": rain_model,
        "temperature_model": temperature_model,
        "features": features,
    }, output_dir / "weather_models.pkl")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("WEATHER MODEL TRAINING COMPLETED", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print("Files saved in declared output folder:", flush=True)
    for file_path in sorted(output_dir.iterdir()):
        if file_path.is_file():
            print(f"{file_path.name} ({file_path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
