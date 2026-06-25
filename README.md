# 🏎️ F1 Podium Predictor

A machine learning project that predicts whether a Formula 1 driver will finish on the podium (top 3) using historical race and qualifying data from 2018 to 2026.

---

## 📌 Overview

This project pulls real F1 session data using the [FastF1](https://theoehrly.github.io/Fast-F1/) library, engineers meaningful features from qualifying and race history, and trains an XGBoost classifier to predict podium finishes. Given a race year and round number, it fetches live qualifying results and outputs the top 3 predicted finishers with confidence percentages.

---

## 📁 Project Structure

```
F1_Podium_prediction/
├── cache/                    # FastF1 cache (auto-generated, not tracked by git)
├── data/
│   ├── quali_results.csv     # Raw qualifying results (2018–2026)
│   ├── race_results.csv      # Raw race results (2018–2026)
│   └── processed_data.csv    # Feature-engineered dataset for modeling
├── model/
│   └── xgb_model.pkl         # Trained XGBoost model
├── data.py                   # Fetches qualifying & race data via FastF1
├── processed_data.py         # Merges data and builds model features
├── train.py                  # Trains and evaluates the XGBoost classifier
├── main.py                   # Predicts podium for a given race
└── analytics.ipynb           # EDA, model evaluation, and model comparison
```

---

## 🔧 Installation

**Prerequisites:** Python 3.8+

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/F1_Podium_prediction.git
   cd F1_Podium_prediction
   ```

2. Install dependencies:
   ```bash
   pip install fastf1 pandas scikit-learn xgboost matplotlib seaborn
   ```

---

## 🚀 Usage

Run the scripts in order:

### 1. Collect Data

Fetches qualifying and race session data for every race from 2018–2026 and saves them as CSVs.

```bash
python data.py
```

> ⚠️ This step makes many API calls and will take a while on first run. FastF1 caches results locally so subsequent runs are much faster.

### 2. Engineer Features

Merges qualifying and race data, then builds predictive features for the model.

```bash
python processed_data.py
```

### 3. Train the Model

Trains an XGBoost classifier on the processed data and saves it to `model/xgb_model.pkl`.

```bash
python train.py
```

### 4. Make a Prediction

Edit the `YEAR` and `ROUND` variables at the top of `main.py`, then run:

```bash
python main.py
```

**Example output:**

```
🏎️  F1 Podium Prediction - Monaco Grand Prix 2026
==================================================
🥇 P1 → VER (Red Bull Racing) - 84% confidence
🥈 P2 → LEC (Scuderia Ferrari) - 71% confidence
🥉 P3 → NOR (McLaren) - 65% confidence
==================================================
```

### 5. Explore Analytics

Open `analytics.ipynb` to view EDA charts, model evaluation, and model comparison visuals.

---

## 🧠 Features Used

| Feature             | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `GridPosition_race` | Starting grid position from qualifying                    |
| `DriverAvgLast3`    | Driver's average finish position in last 3 races          |
| `TrackHistoryAvg`   | Driver's historical average finish at this specific track |
| `TeamAvgSeason`     | Team's average finish position so far this season         |
| `ChampionshipPos`   | Driver's championship standing before the race            |

---

## 🤖 Model

- **Algorithm:** XGBoost Classifier
- **Target:** Binary — `1` if the driver finishes P1/P2/P3, `0` otherwise
- **Class imbalance:** Handled via `scale_pos_weight=6` (podiums are rare — only 3 out of ~20 drivers per race finish on the podium)
- **Train/test split:** 80/20

### Results

Overall accuracy is **86%**, however this metric is misleading due to class imbalance (~15% of drivers podium per race). The more meaningful metrics are:

| Metric           | XGBoost | Logistic Regression (baseline) |
| ---------------- | ------- | ------------------------------ |
| Accuracy         | 86%     | 81%                            |
| Podium Precision | 51%     | 42%                            |
| Podium Recall    | 79%     | 82%                            |
| Podium F1 Score  | 0.62    | 0.55                           |

XGBoost outperforms the baseline on almost every metric, justifying the use of a more complex algorithm.

**Cross Validation (5-fold):**

- Mean Accuracy: 87.7%
- Std Deviation: 0.008

The low standard deviation confirms the model generalizes consistently and is not overfitting to a single train/test split.

---

## 📊 Analytics

`analytics.ipynb` includes:

- Podium rate by grid position (EDA)
- Feature importance plot
- Confusion matrix heatmap
- Classification report
- XGBoost vs Logistic Regression comparison chart

---

## 📦 Dependencies

| Package        | Purpose                                   |
| -------------- | ----------------------------------------- |
| `fastf1`       | Fetching official F1 session data         |
| `pandas`       | Data manipulation and feature engineering |
| `scikit-learn` | Train/test splitting and model evaluation |
| `xgboost`      | Gradient boosted classifier               |
| `matplotlib`   | Plotting and visualizations               |
| `seaborn`      | Confusion matrix heatmap                  |
| `pickle`       | Saving and loading the trained model      |

---

## ⚠️ Notes

- Predictions for **Round 1** of any season default missing historical features to a neutral value of `10` (mid-grid), since no prior season data exists yet.
- Drivers with no prior history at a specific track also receive the neutral fallback.
- The model is trained on completed races only — sprint results are not included.

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
