# ⚙️ AI-Based Manufacturing Efficiency Classification Dashboard

> Real-Time Predictive Intelligence Using Industrial IoT Sensor, Production Quality, and 6G Network Telemetry

An interactive Streamlit dashboard that classifies manufacturing efficiency states (**High**, **Medium**, **Low**) across a fleet of 50 industrial machines using 100,000 telemetry records. It combines sensor data, production metrics, and 6G network quality indicators with three ML classifiers and real-time explainability.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📑 Table of Contents

- [Features](#-features)
- [Dashboard Modules](#-dashboard-modules)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Engineered Features](#-engineered-features)
- [Models & Metrics](#-models--metrics)
- [Screenshots](#-screenshots)
- [License](#-license)

---

## ✨ Features

- **Three ML classifiers** — Logistic Regression (baseline), Random Forest, and Gradient Boosting — trained, cached, and benchmarked side-by-side.
- **Live inference simulator** — adjust 10+ sensor/network parameters in real time and see the predicted efficiency class with confidence scores.
- **Sensitivity stress-testing** — production speed and error rate multiplier sliders for "what-if" scenario analysis.
- **Global & local explainability (XAI)** — feature importance rankings and per-prediction attribution with directional impact labelling.
- **Per-machine profiling** — fleet-wide efficiency distribution, individual machine historical trend lines, and top-10 leaderboard.
- **6G network monitoring** — latency/packet-loss filters, network reliability scores, and cross-metric comparison tables.
- **Responsive KPI ribbon** — top-level metrics update instantly as filters change.

---

## 🧩 Dashboard Modules

| # | Module | Description |
|---|--------|-------------|
| 1 | **Efficiency Prediction Dashboard** | Real-time classification simulator with confidence bar chart and model benchmark comparison table + confusion matrix. |
| 2 | **Machine-Level Insights** | Stacked efficiency bars for the full fleet, individual machine daily trend, and top-10 high-efficiency leaderboard. |
| 3 | **Explainability Panel (XAI)** | Global feature importance (top 10) and local contribution waterfall for the current simulated state. |
| 4 | **Operational & 6G Monitoring** | Operation mode breakdown, production-speed vs. error-rate phase-space scatter, and aggregated metric comparison table across efficiency states. |

---

## 🛠 Tech Stack

| Component | Library |
|-----------|---------|
| Web framework | [Streamlit](https://streamlit.io/) ≥ 1.28 |
| Data manipulation | [Pandas](https://pandas.pydata.org/) ≥ 2.0, [NumPy](https://numpy.org/) ≥ 1.24 |
| Visualisation | [Plotly](https://plotly.com/python/) ≥ 5.18 |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) ≥ 1.3 |

---

## 📊 Dataset

The dashboard expects a CSV file named **`Thales_Group_Manufacturing.csv`** with the following columns:

| Column | Description |
|--------|-------------|
| `Date` | Record date (`DD-MM-YYYY`) |
| `Timestamp` / `Time` | Record time (`HH:MM:SS`) |
| `Machine_ID` | Machine identifier (1–50) |
| `Temperature_C` | Sensor temperature in °C |
| `Vibration_Hz` | Vibration frequency in Hz |
| `Power_Consumption_kW` | Power draw in kW |
| `Network_Latency_ms` | 6G network latency in ms |
| `Packet_Loss_%` | Network packet loss percentage |
| `Quality_Control_Defect_Rate_%` | QC defect rate |
| `Production_Speed_units_per_hr` | Production throughput |
| `Predictive_Maintenance_Score` | Maintenance readiness (0–1) |
| `Error_Rate_%` | Operational error rate |
| `Operation_Mode` | `Active` / `Idle` / `Maintenance` |
| `Efficiency_Status` | Target label — `High` / `Medium` / `Low` |

Place the CSV in the **project root** directory (same folder as the `.py` file).  
Alternatively, upload any CSV with the same schema via the sidebar file uploader at runtime.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/thales-manufacturing-dashboard.git
cd thales-manufacturing-dashboard

# 2. Create and activate a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run

```bash
streamlit run streamlit_manufacturing_dashboard.py
```

The dashboard opens automatically at **http://localhost:8501**.

---

## 🎮 Usage

1. **Upload data** — Use the sidebar file uploader or place `Thales_Group_Manufacturing.csv` in the project root.
2. **Select a model** — Choose between Gradient Boosting, Random Forest, or Logistic Regression.
3. **Filter the fleet** — Narrow by operation mode, specific machines, date range, and 6G network quality thresholds.
4. **Simulate live telemetry** — Adjust sensor and network inputs in Module 1 to get instant efficiency predictions.
5. **Stress-test with sensitivity sliders** — Modify the error and speed multipliers to explore edge-case scenarios.
6. **Explore explainability** — Switch to the XAI tab to understand which features drive predictions globally and locally.

---

## 📁 Project Structure

```
thales-manufacturing-dashboard/
├── streamlit_manufacturing_dashboard.py   # Main Streamlit application
├── Thales_Group_Manufacturing.csv         # Dataset (100K records, 50 machines)
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## 🔧 Engineered Features

The dashboard automatically engineers four composite features from raw telemetry:

| Feature | Formula | Purpose |
|---------|---------|---------|
| **Sensor Stability Index** | `100 × (1 − 0.5 × T_norm − 0.5 × V_norm)` | Combined thermal & vibration health (0–100) |
| **Energy Efficiency Ratio** | `Production_Speed / Power_Consumption` | Units produced per kW |
| **Error-to-Output Ratio** | `Error_Rate × 100 / Production_Speed` | Error burden relative to throughput |
| **Network Reliability Score** | `100 × (1 − 0.5 × Lat_norm − 0.5 × PL_norm)` | 6G network quality (0–100) |

---

## 📈 Models & Metrics

| Model | Role | Key Config |
|-------|------|------------|
| **Logistic Regression** | Baseline | `class_weight="balanced"`, standardised inputs |
| **Random Forest** | Advanced A | 120 trees, max depth 12, class-balanced |
| **Gradient Boosting** | Advanced B | 100 estimators, learning rate 0.1, max depth 5 |

All models are evaluated on a stratified 80/20 train-test split. Reported metrics include:
Accuracy, Balanced Accuracy, Macro F1, Weighted F1, and Mean Prediction Confidence.

---

## 📸 Screenshots

> _Add screenshots of each dashboard tab here._

---

## 📄 License

This project is provided under the [MIT License](LICENSE). Feel free to use, modify, and distribute.

---

<p align="center">
  Built with ❤️ using <a href="https://streamlit.io/">Streamlit</a> and <a href="https://scikit-learn.org/">scikit-learn</a>
</p>
