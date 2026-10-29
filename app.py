"""
Weather IoT Dashboard
---------------------
A simple UI for the rain / next-temperature models trained in weather_train.py.

Run it from VS Code:
    1. Open a terminal in this folder (the one with app.py, weather_models.pkl, weather_iot_dataset.csv)
    2. pip install streamlit pandas scikit-learn joblib matplotlib
    3. streamlit run app.py

It will open in your browser at http://localhost:8501
"""

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Weather IoT Dashboard", page_icon="🌦️", layout="centered")

HERE = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    "hour", "day_of_year", "temperature_c", "humidity_pct",
    "pressure_hpa", "wind_speed_kmh", "recent_rainfall_mm",
]

FEATURE_INFO = {
    "hour": dict(label="Hour of day", min=0, max=23, default=12, step=1),
    "day_of_year": dict(label="Day of year", min=1, max=365, default=180, step=1),
    "temperature_c": dict(label="Current temperature (°C)", min=-10.0, max=45.0, default=25.0, step=0.1),
    "humidity_pct": dict(label="Humidity (%)", min=0.0, max=100.0, default=65.0, step=0.1),
    "pressure_hpa": dict(label="Pressure (hPa)", min=980.0, max=1040.0, default=1013.0, step=0.1),
    "wind_speed_kmh": dict(label="Wind speed (km/h)", min=0.0, max=60.0, default=10.0, step=0.1),
    "recent_rainfall_mm": dict(label="Recent rainfall (mm)", min=0.0, max=50.0, default=1.0, step=0.1),
}


@st.cache_resource
def load_models():
    """Load the combined models file, falling back to separate files if needed."""
    combined_path = os.path.join(HERE, "weather_models.pkl")
    if os.path.exists(combined_path):
        bundle = joblib.load(combined_path)
        return bundle["rain_model"], bundle["temperature_model"], bundle.get("features", FEATURES)

    rain_path = os.path.join(HERE, "rain_model.pkl")
    temp_path = os.path.join(HERE, "temperature_model.pkl")
    if os.path.exists(rain_path) and os.path.exists(temp_path):
        return joblib.load(rain_path), joblib.load(temp_path), FEATURES

    return None, None, FEATURES


@st.cache_data
def load_dataset():
    csv_path = os.path.join(HERE, "weather_iot_dataset.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def main():
    st.title("🌦️ Weather IoT Dashboard")
    st.caption("Predict next-step rain and next temperature from current sensor readings.")

    rain_model, temperature_model, features = load_models()

    if rain_model is None or temperature_model is None:
        st.error(
            "Couldn't find model files. Place **weather_models.pkl** "
            "(or both rain_model.pkl and temperature_model.pkl) in the same "
            "folder as app.py."
        )
        st.stop()

    tab_predict, tab_data = st.tabs(["🔮 Predict", "📊 Dataset"])

    with tab_predict:
        st.subheader("Current readings")
        cols = st.columns(2)
        values = {}
        for i, feat in enumerate(features):
            info = FEATURE_INFO.get(feat, dict(label=feat, min=0.0, max=100.0, default=0.0, step=0.1))
            with cols[i % 2]:
                if isinstance(info["default"], int):
                    values[feat] = st.number_input(
                        info["label"], min_value=int(info["min"]), max_value=int(info["max"]),
                        value=int(info["default"]), step=int(info["step"]), key=feat,
                    )
                else:
                    values[feat] = st.slider(
                        info["label"], min_value=float(info["min"]), max_value=float(info["max"]),
                        value=float(info["default"]), step=float(info["step"]), key=feat,
                    )

        st.divider()

        if st.button("Predict", type="primary", use_container_width=True):
            X = pd.DataFrame([[values[f] for f in features]], columns=features)

            rain_pred = rain_model.predict(X)[0]
            rain_proba = (
                rain_model.predict_proba(X)[0][1]
                if hasattr(rain_model, "predict_proba")
                else None
            )
            next_temp = temperature_model.predict(X)[0]

            c1, c2 = st.columns(2)
            with c1:
                if rain_pred == 1:
                    st.metric("Rain next step", "🌧️ Yes")
                else:
                    st.metric("Rain next step", "☀️ No")
                if rain_proba is not None:
                    st.progress(float(rain_proba), text=f"Rain probability: {rain_proba:.1%}")
            with c2:
                delta = next_temp - values["temperature_c"]
                st.metric("Next temperature", f"{next_temp:.1f} °C", delta=f"{delta:+.1f} °C")

    with tab_data:
        df = load_dataset()
        if df is None:
            st.info("weather_iot_dataset.csv not found in this folder — dataset preview unavailable.")
        else:
            st.write(f"**{len(df):,} rows** · columns: {', '.join(df.columns)}")
            st.dataframe(df.head(200), use_container_width=True, height=300)

            st.subheader("Temperature over the dataset")
            st.line_chart(df["temperature_c"].reset_index(drop=True))

            st.subheader("Rain events (rain_next_step)")
            rain_rate = df["rain_next_step"].mean()
            st.write(f"Rain occurs in **{rain_rate:.1%}** of recorded steps.")
            st.bar_chart(df["rain_next_step"].value_counts().sort_index())


if __name__ == "__main__":
    main()
