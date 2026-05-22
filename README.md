# 🏎️ F1 Podium Predictor

A machine learning project that predicts whether a Formula 1 driver will finish on the podium (top 3) using historical race and qualifying data from 2018 to 2026.

---

## 📌 Overview

This project pulls real F1 session data using the [FastF1](https://theoehrly.github.io/Fast-F1/) library, engineers meaningful features from qualifying and race history, and trains an XGBoost classifier to predict podium finishes. Given a race year and round number, it fetches live qualifying results and outputs the top 3 predicted finishers with confidence percentages.

---

## 📁 Project Structure

```
F1_podium/
├── cache/                    # FastF1 cache (auto-generated)
├── data/
│   ├── quali_results.csv     # Raw qualifying results (2018–2026)
│   ├── race_results.csv      # Raw race results (2018–2026)
│   └── processed_data.csv    # Feature-engineered dataset for modeling
├── model/
│   └── xgb_model.pkl         # Trained XGBoost model
├── data_collection.py        # Fetches qualifying & race data via FastF1
├── feature_engineering.py    # Merges data and builds model features
├── train_model.py            # Trains and evaluates the XGBoost classifier
└── predict.py                # Predicts podium for a given race
```

---

## 🔧 Installation

**Prerequisites:** Python 3.8+

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/F1_podium.git
   cd F1_podium
   ```

2. Install dependencies:
   ```bash
   pip install fastf1 pandas scikit-learn xgboost
   ```

---

## 🚀 Usage

Run the scripts in order:

### 1. Collect Data
Fetches qualifying and race session data for every race from 2018–2026 and saves them as CSVs.
```bash
python data_collection.py
```
> ⚠️ This step makes many API calls and will take a while on first run. FastF1 caches results locally so subsequent runs are much faster.

### 2. Engineer Features
Merges qualifying and race data, then builds predictive features for the model.
```bash
python feature_engineering.py
```

### 3. Train the Model
Trains an XGBoost classifier on the processed data and saves it to `model/xgb_model.pkl`.
```bash
python train_model.py
```

### 4. Make a Prediction
Edit the `YEAR` and `ROUND` variables at the top of `predict.py`, then run:
```bash
python predict.py
```

**Example output:**
```
🏎️  F1 Podium Prediction - Bahrain Grand Prix 2025
==================================================
🥇 P1 → VER (Red Bull Racing) - 84% confidence
🥈 P2 → LEC (Scuderia Ferrari) - 71% confidence
🥉 P3 → NOR (McLaren) - 65% confidence
==================================================
```

---

## 🧠 Features Used

| Feature | Description |
|---|---|
| `GridPosition_race` | Starting grid position from qualifying |
| `DriverAvgLast3` | Driver's average finish position in last 3 races |
| `TrackHistoryAvg` | Driver's historical average finish at this specific track |
| `TeamAvgSeason` | Team's average finish position so far this season |
| `ChampionshipPos` | Driver's championship standing before the race |

---

## 🤖 Model

- **Algorithm:** XGBoost Classifier
- **Target:** Binary — `1` if the driver finishes P1/P2/P3, `0` otherwise
- **Class imbalance:** Handled via `scale_pos_weight=6` (podiums are rare — only 3 out of ~20 drivers per race)
- **Train/test split:** 80/20

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `fastf1` | Fetching official F1 session data |
| `pandas` | Data manipulation and feature engineering |
| `scikit-learn` | Train/test splitting and model evaluation |
| `xgboost` | Gradient boosted classifier |
| `pickle` | Saving and loading the trained model |

---

## ⚠️ Notes

- Predictions for **Round 1** of any season default missing historical features to a neutral value of `10` (mid-grid), since no prior season data exists yet.
- Drivers with no prior history at a specific track also receive the neutral fallback.
- The model is trained on completed races only — sprint results are not included.

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
