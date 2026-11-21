# 🏞️ Nilgiris Spatio-Temporal Climate Prediction & Early Warning System

An intelligent climate intelligence dashboard and machine learning forecast platform tailored for the unique micro-climates of **The Nilgiris** (Western Ghats, Tamil Nadu: Ooty, Coonoor, Kotagiri, Gudalur, and Wellington).

Built using **Python**, **Streamlit**, **LightGBM**, and **Plotly**.

---

## 📌 Project Overview

Due to extreme topography and elevation variance, the Nilgiris biosphere experiences localized micro-climates requiring hyper-local climate models. This application provides:
1. **Live Atmospheric Monitoring**: Real-time integration with Open-Meteo weather APIs for live temperature, precipitation, and wind speeds.
2. **Proactive Hazard Detection**: Real-time evaluation of Heat, High Wind, and Heavy Rain hazards paired with browser push notifications.
3. **Machine Learning Autoregressive Forecasting**: LightGBM models trained on historical time-series data to deliver sequential 24-hour future temperature trajectories.
4. **Interactive 3D Visualizations**: Multi-dimensional trajectory charts and embedded live weather radar.

---

## 🚀 Key Features

- **📍 Micro-Climate Location Selection**: Monitor hyper-local weather across Nilgiris locations:
  - **Ooty** *(11.4102° N, 76.6950° E)*
  - **Coonoor** *(11.3536° N, 76.7937° E)*
  - **Kotagiri** *(11.4268° N, 76.8838° E)*
  - **Gudalur** *(11.4167° N, 76.6833° E)*
  - **Wellington** *(11.3653° N, 76.7869° E)*
- **🌡️ Live Environmental Metrics**: Instant tracking of current ambient temperature, wind velocity, and WMO weather codes.
- **🚨 Proactive Early Warning System**:
  - Automatically identifies **Heat Watch**, **Wind Watch**, and **Rain Watch** conditions.
  - Injects native browser desktop notifications when safety thresholds are exceeded.
- **🤖 24-Hour 3D LightGBM Forecast**:
  - Uses an autoregressive sequence (72 hours context window) to predict 24 hours into the future.
  - Automatically classifies forecast trend as *warming*, *cooling*, or *stable*.
  - Renders interactive 3D trajectories with Plotly.
- **🗺️ Interactive Live Radar**: Embedded Windy atmospheric radar showing real-time wind and cloud dynamics.
- **⚡ Automated Model Self-Training**: If model weights are missing, the system automatically fetches historical weather data and trains a custom LightGBM regressor on the fly.

---

## 🛠️ Tools & Technologies

- **Language**: Python 3.x
- **Web Dashboard**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [LightGBM](https://lightgbm.readthedocs.io/) (`LGBMRegressor`, `early_stopping`)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Interactive Visualizations**: [Plotly Graph Objects](https://plotly.com/python/) (3D Surface/Scatter)
- **Data Normalization & Serialization**: [Scikit-learn](https://scikit-learn.org/) (`MinMaxScaler`), [Joblib](https://joblib.readthedocs.io/)
- **External APIs**: [Open-Meteo Weather APIs](https://open-meteo.com/) (Live Forecast & Historical Archive)

---

## 📁 Repository Structure

```text
├── app.py                     # Streamlit application & interactive dashboard
├── model_trainer.py           # Standalone model training CLI utility
├── coonoor_model.txt          # Pre-trained LightGBM model for Coonoor
├── coonoor_scaler.gz          # Feature scaler for Coonoor
├── ooty_model.txt             # Pre-trained LightGBM model for Ooty
├── ooty_scaler.gz             # Feature scaler for Ooty
├── kotagiri_model.txt         # Pre-trained LightGBM model for Kotagiri
├── kotagiri_scaler.gz         # Feature scaler for Kotagiri
├── gudalur_model.txt          # Pre-trained LightGBM model for Gudalur
├── gudalur_scaler.gz          # Feature scaler for Gudalur
├── wellington_model.txt       # Pre-trained LightGBM model for Wellington
├── wellington_scaler.gz       # Feature scaler for Wellington
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Comprehensive project documentation
```

---

## 🧪 Installation & How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Ashmitha-ash/climate-prediction-nilgiris.git
cd climate-prediction-nilgiris
```

### 2. (Optional) Create and activate a virtual environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```

### 4. (Optional) Train or update a location model manually
```bash
python model_trainer.py --location Ooty
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 👤 Author

- **Ashmitha A** – B.Sc. Data Science
