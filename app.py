import streamlit as st
import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import os
import time
import plotly.graph_objects as go
import warnings

# Suppress harmless warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
APP_TITLE = "Spatio-Temporal Climate Prediction"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NILGIRIS_LOCATIONS = {
    "Coonoor": {"lat": 11.3536, "lon": 76.7937},
    "Ooty": {"lat": 11.4102, "lon": 76.6950},
    "Kotagiri": {"lat": 11.4268, "lon": 76.8838},
    "Gudalur": {"lat": 11.4167, "lon": 76.6833},
    "Wellington": {"lat": 11.3653, "lon": 76.7869}
}
SEQUENCE_LENGTH = 72

# --- ======================================================================= ---
# --- DATA FETCHING (UNCached - for reliability) ---
# --- ======================================================================= ---
def fetch_data_from_api(lat, lon):
    """Makes a direct API call to get live and historical data."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=precipitation,windspeed_10m"
        response = requests.get(url, timeout=15); response.raise_for_status()
        live_data = response.json()
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        hist_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m"
        hist_response = requests.get(hist_url, timeout=15); hist_response.raise_for_status()
        df_recent = pd.DataFrame(hist_response.json()['hourly'])
        df_recent['time'] = pd.to_datetime(df_recent['time'])
        return live_data, df_recent.dropna()
    except requests.exceptions.RequestException as e:
        st.error(f"Network Error: Could not connect to weather API. Details: {e}")
        return None, None

# --- ## NEW ## Browser Notification Function ---
def trigger_browser_notification(title, body):
    """Injects JavaScript to show a browser notification."""
    js_code = f"""
        <script>
        if (!("Notification" in window)) {{
            console.log("This browser does not support desktop notification");
        }} else if (Notification.permission === "granted") {{
            new Notification("{title}", {{ body: "{body}", icon: "🏞️" }});
        }} else if (Notification.permission !== "denied") {{
            Notification.requestPermission().then(function (permission) {{
                if (permission === "granted") {{
                    new Notification("{title}", {{ body: "{body}", icon: "🏞️" }});
                }}
            }});
        }}
        </script>
    """
    components.html(js_code, height=0, width=0)

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🏞️", layout="wide")
    st.markdown("""
    <style>
        .stApp { background-color: #1a202c; color: #e2e8f0; }
        [data-testid="stMetric"] { background-color: #2d3748; border-radius: 0.5rem; padding: 1rem; }
        h1, h2, h3 { color: #ffffff; }
        .nlp-summary {
            background-color: rgba(45, 55, 72, 0.7); border-radius: 0.5rem;
            padding: 1.5rem; margin-top: 1rem; font-size: 1.1rem; line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title(f"🏞️ {APP_TITLE}")

    # Initialize session state for caching and notifications
    if 'data_cache' not in st.session_state: st.session_state.data_cache = {}
    if 'last_location' not in st.session_state: st.session_state.last_location = ""
    if 'notifications_sent' not in st.session_state: st.session_state.notifications_sent = {}


    col1, col2 = st.columns([3,1])
    with col1:
        selected_location_name = st.selectbox(
            "Select a Location in The Nilgiris to Monitor:",
            options=list(NILGIRIS_LOCATIONS.keys())
        )
    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        force_refresh = st.button("🔄 Refresh Data")

    location_data = NILGIRIS_LOCATIONS[selected_location_name]
    model_filename = os.path.join(BASE_DIR, f"{selected_location_name.lower()}_model.txt")
    scaler_filename = os.path.join(BASE_DIR, f"{selected_location_name.lower()}_scaler.gz")

    if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
        run_training_process(selected_location_name, location_data['lat'], location_data['lon'], model_filename, scaler_filename)
    else:
        if selected_location_name != st.session_state.last_location or force_refresh:
            with st.spinner(f"Fetching live weather data for {selected_location_name}..."):
                live_data, df_recent = fetch_data_from_api(location_data['lat'], location_data['lon'])
                st.session_state.data_cache[selected_location_name] = (live_data, df_recent)
                st.session_state.last_location = selected_location_name
                st.session_state.notifications_sent = {} # Reset notifications on refresh/location change
        
        cached_data = st.session_state.data_cache.get(selected_location_name)
        if cached_data:
            run_live_app(selected_location_name, location_data['lat'], location_data['lon'], model_filename, scaler_filename, cached_data)
        else:
            with st.spinner(f"Fetching initial data for {selected_location_name}..."):
                live_data, df_recent = fetch_data_from_api(location_data['lat'], location_data['lon'])
                st.session_state.data_cache[selected_location_name] = (live_data, df_recent)
                st.session_state.last_location = selected_location_name
                st.rerun()

# --- ONE-TIME MODEL TRAINING ---
def run_training_process(location_name, lat, lon, model_filename, scaler_filename):
    # This function is unchanged
    from sklearn.preprocessing import MinMaxScaler
    st.title(f"🛠️ One-Time Model Setup for {location_name}"); st.info(f"The AI model for {location_name} was not found. Starting automatic training."); st.warning("This will take about 60 seconds and only happens once for this location.")
    with st.spinner(f"Fetching historical data..."):
        API_URL = "https://archive-api.open-meteo.com/v1/archive"; params = { "latitude": lat, "longitude": lon, "start_date": "2022-01-01", "end_date": datetime.now().strftime("%Y-%m-%d"), "hourly": "temperature_2m" }; response = requests.get(API_URL, params=params, timeout=60); response.raise_for_status()
        df = pd.DataFrame(response.json()['hourly']); df['time'] = pd.to_datetime(df['time']); dataframe = df[['temperature_2m']].dropna()
    st.success(f"✅ Download complete.")
    progress_bar = st.progress(0, text="Training model..."); scaler = MinMaxScaler(feature_range=(0, 1)); scaled_data = scaler.fit_transform(dataframe)
    X, y = [], [];
    for i in range(len(scaled_data) - SEQUENCE_LENGTH): X.append(scaled_data[i:(i + SEQUENCE_LENGTH)].flatten()); y.append(scaled_data[i + SEQUENCE_LENGTH])
    X, y = np.array(X), np.array(y).ravel()
    lgb_params = {'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000, 'learning_rate': 0.05, 'verbose': -1}; model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X, y, eval_set=[(X, y)], eval_metric='mae', callbacks=[lgb.early_stopping(10, verbose=False)]); progress_bar.progress(1.0, text="Training Complete!")
    model.booster_.save_model(model_filename); joblib.dump(scaler, scaler_filename)
    st.success(f"✅ Model training for {location_name} finished!"); st.info("The model is saved. The application will now reload."); time.sleep(5); st.rerun()

# --- LIVE APPLICATION ---
def run_live_app(location_name, lat, lon, model_filename, scaler_filename, cached_data):
    st.subheader(f"A Proactive Climate Detector for {location_name}, Nilgiris")
    st.write(f"**Data as of:** {datetime.now().strftime('%I:%M:%S %p')}")

    @st.cache_resource
    def load_model_and_scaler(_model_file, _scaler_file):
        model = lgb.Booster(model_file=_model_file)
        scaler = joblib.load(_scaler_file)
        return model, scaler

    model, scaler = load_model_and_scaler(model_filename, scaler_filename)
    live_data, df_recent = cached_data

    if live_data and df_recent is not None:
        current = live_data['current_weather']
        tab1, tab2, tab3, tab4 = st.tabs(["Live Metrics", "Alerts Dashboard", "3D ML Forecast", "Live Radar Map"])
        
        with tab1:
            st.header(f"Real-Time Conditions in {location_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Temperature", f"{current['temperature']}°C")
            col2.metric("Wind Speed", f"{current['windspeed']} km/h", "💨")
            col3.metric("Weather Code", f"{current['weathercode']}", "ℹ️")
        with tab2:
            st.header(f"Proactive Alerts for {location_name}")
            col1, col2, col3 = st.columns(3)
            # ## NEW ## Notification logic integrated with alerts
            with col1:
                temp = current['temperature']
                if temp > 30:
                    st.metric("Heat Status", "Watch 🟡", f"{temp}°C")
                    if not st.session_state.notifications_sent.get('heat'):
                        trigger_browser_notification(f"Heat Watch for {location_name}", f"Temperature has reached {temp}°C.")
                        st.session_state.notifications_sent['heat'] = True
                else:
                    st.metric("Heat Status", "Safe ✅", f"{temp}°C")
                    st.session_state.notifications_sent['heat'] = False
            with col2:
                wind = current['windspeed']
                if wind > 40:
                    st.metric("Wind Status", "Watch 🟡", f"{wind} km/h")
                    if not st.session_state.notifications_sent.get('wind'):
                        trigger_browser_notification(f"Wind Watch for {location_name}", f"Wind speed has reached {wind} km/h.")
                        st.session_state.notifications_sent['wind'] = True
                else:
                    st.metric("Wind Status", "Safe ✅", f"{wind} km/h")
                    st.session_state.notifications_sent['wind'] = False
            with col3:
                max_precip = max(live_data['hourly']['precipitation'][0:12])
                if max_precip > 2:
                    st.metric("Rain Status", "Watch 🟡", f"{max_precip}mm/h")
                    if not st.session_state.notifications_sent.get('rain'):
                        trigger_browser_notification(f"Rain Watch for {location_name}", f"Rain forecasted at {max_precip} mm/h.")
                        st.session_state.notifications_sent['rain'] = True
                else:
                    st.metric("Rain Status", "Safe ✅", f"{max_precip}mm/h")
                    st.session_state.notifications_sent['rain'] = False
        with tab3:
            st.header(f"ML 3D Forecast for {location_name}")
            with st.spinner('Running model...'):
                last_sequence_raw = df_recent['temperature_2m'].tail(SEQUENCE_LENGTH).values
                last_sequence_scaled = scaler.transform(last_sequence_raw.reshape(-1, 1))
                predictions = []; current_input = last_sequence_scaled.flatten().tolist()
                for _ in range(24):
                    pred_scaled = model.predict([current_input])[0]
                    predictions.append(pred_scaled)
                    current_input = current_input[1:] + [pred_scaled]
                predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            st.subheader("NLP Summary")
            max_temp, min_temp = np.max(predictions), np.min(predictions)
            trend = "stable"
            if predictions[-1] > predictions[0] + 1.5: trend = "warming"
            elif predictions[-1] < predictions[0] - 1.5: trend = "cooling"
            st.markdown(f'<div class="nlp-summary">The model predicts a {trend} trend, with a high of **{max_temp:.1f}°C** and a low of **{min_temp:.1f}°C**.</div>', unsafe_allow_html=True)
            
            st.subheader("3D Forecast Trajectory")

            # ## MODIFIED ## Added data labels to the 3D chart
            # Create text labels for each data point
            text_labels = [f'{p:.1f}°' for p in predictions]

            fig = go.Figure(data=go.Scatter3d(
                x=np.arange(24), 
                y=predictions, 
                z=np.sin(np.linspace(0,2*np.pi,24))*5,
                mode='lines+markers+text',  # Add markers and text to the mode
                text=text_labels,          # Add the text labels
                textposition='top center',
                textfont=dict(color='white', size=10),
                marker=dict(
                    size=4,
                    color=predictions,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                line=dict(color=predictions, colorscale='Viridis', width=8)
            ))
            fig.update_layout(
                title='Temperature Forecast', 
                scene=dict(
                    xaxis_title='Hours from Now', 
                    yaxis_title='Temp (°C)', 
                    zaxis_title='Pressure (Sim)'
                ), 
                template='plotly_dark', 
                margin=dict(l=0,r=0,b=0,t=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.header(f"Radar Map for {location_name}")
            components.iframe(f"https://embed.windy.com/embed.html?lat={lat}&lon={lon}&zoom=11", height=600)

if __name__ == "__main__":
    main()


