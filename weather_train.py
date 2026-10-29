import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

data = pd.read_csv("weather_iot_dataset.csv")

os.makedirs("outputs", exist_ok=True)

# Model 1: Rain Prediction
X = data[["hour", "day_of_year", "temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh", "recent_rainfall_mm"]]
y = data["rain_next_step"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
rain_model.fit(X_train, y_train)

predictions = rain_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Rain Prediction Accuracy:", accuracy)

joblib.dump(rain_model, "outputs/rain_model.pkl")

print("Rain model saved as rain_model.pkl")

# Model 2: Temperature Prediction
X = data[["hour", "day_of_year", "temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh", "recent_rainfall_mm"]]
y = data["next_temperature_c"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

temperature_model = RandomForestRegressor(n_estimators=100, random_state=42)
temperature_model.fit(X_train, y_train)

predictions = temperature_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Next Temperature MAE:", mae)

joblib.dump(temperature_model, "outputs/temperature_model.pkl")

print("Temperature model saved as temperature_model.pkl")

# Save both models together
joblib.dump(
    {
        "rain_model": rain_model,
        "temperature_model": temperature_model,
        "features": X.columns.tolist()
    },
    "outputs/weather_models.pkl"
)

print("Combined model saved as weather_models.pkl")
print("Training completed successfully.")
print("Files saved:", os.listdir("outputs"))