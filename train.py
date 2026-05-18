import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import pandas as pd

# Load the processed data
data = pd.read_csv('/Users/rishitsingh/Desktop/F1_podium/data/processed_data.csv')

# Define features and target variable
X = data[['GridPosition_race', 'DriverAvgLast3', 'TrackHistoryAvg', 'TeamAvgSeason', 'ChampionshipPos']]
y = data['Podium']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost Classifier
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=6  # ← tells model podiums are rare
)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

os.makedirs('/Users/rishitsingh/Desktop/F1_podium/model', exist_ok=True)

# Save the trained model to a file
with open('/Users/rishitsingh/Desktop/F1_podium/model/xgb_model.pkl', 'wb') as file:
    pickle.dump(model, file)  