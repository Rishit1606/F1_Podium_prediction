import fastf1
import pandas as pd
import pickle
import os


# change these to predict any race

YEAR  = 2018
ROUND = 3  # Japan GP


# Load the trained model

with open('/Users/rishitsingh/Desktop/F1_podium/model/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
    print(model.get_booster().feature_names)

# Load processed data
historical_df = pd.read_csv('/Users/rishitsingh/Desktop/F1_podium/data/processed_data.csv')


fastf1.Cache.enable_cache('/Users/rishitsingh/Desktop/F1_podium/cache')

# Fetch qualifying session for the specified race
quali = fastf1.get_session(YEAR, ROUND, 'Q')
quali.load(laps=False, telemetry=False, weather=False, messages=False)

predict_df = quali.results[['DriverNumber', 'Abbreviation', 'TeamName', 'Position']].copy()
predict_df = predict_df.rename(columns={'Position': 'GridPosition_race'})
predict_df['Year']  = YEAR
predict_df['Round'] = ROUND

# Get event name
schedule   = fastf1.get_event_schedule(YEAR, include_testing=False)
Event_name = schedule[schedule['RoundNumber'] == ROUND]['EventName'].item()
predict_df['EventName'] = Event_name

print(f"✅ Qualifying data loaded for {Event_name} {YEAR}")

# Get the Driver's average grid position in the last 3 races before this one in the same season
races_before = historical_df[(historical_df['Year'] == YEAR) & (historical_df['Round'] < ROUND)]
driver_avg = races_before.groupby('Abbreviation')['Position_race'].apply(lambda x: x.tail(3).mean()).reset_index()
driver_avg.columns = ['Abbreviation', 'DriverAvgLast3']
predict_df = predict_df.merge(driver_avg, on='Abbreviation', how='left')

# Get the Driver's average grid position at this track in all previous races   
track_history = historical_df[(historical_df['EventName'] == Event_name) & (historical_df['Year'] < YEAR)].groupby('Abbreviation')['Position_race'].mean().reset_index()
track_history.columns = ['Abbreviation', 'TrackHistoryAvg']
predict_df = predict_df.merge(track_history, on='Abbreviation', how='left')

# Get the Team's average grid position this season before this race
team_avg = races_before.groupby('TeamName')['Position_race'].mean().reset_index()
team_avg.columns = ['TeamName', 'TeamAvgSeason']
predict_df = predict_df.merge(team_avg, on='TeamName', how='left')

# Calculate ChampionshipPos
champ = historical_df[(historical_df['Year'] == YEAR) & (historical_df['Round'] < ROUND)].groupby('Abbreviation')['Points_race'].sum().reset_index()
champ.columns = ['Abbreviation', 'ChampionshipPos']
champ['ChampionshipPos'] = champ['ChampionshipPos'].rank(ascending=False, method='min')
predict_df = predict_df.merge(champ[['Abbreviation', 'ChampionshipPos']], on='Abbreviation', how='left')

# Fill missing values
predict_df = predict_df.fillna(10)

print("\n📊 Features for prediction:")
print(predict_df[['Abbreviation', 'GridPosition_race', 'DriverAvgLast3', 'TrackHistoryAvg', 'TeamAvgSeason', 'ChampionshipPos']])

# Make Prediction
features = ['GridPosition_race', 'DriverAvgLast3', 'TrackHistoryAvg', 'TeamAvgSeason', 'ChampionshipPos']
predict_df[features] = predict_df[features].astype(float)
predict_df['PodiumProbability'] = model.predict_proba(predict_df[features].values)[:, 1]

# Sort by probability and get top 3
podium = predict_df.nlargest(3, 'PodiumProbability')

# Displaying Result
print(f"\n🏎️  F1 Podium Prediction - {Event_name} {YEAR}")
print("=" * 50)
medals = ['🥇', '🥈', '🥉']
for i, (_, driver) in enumerate(podium.iterrows()):
    print(f"{medals[i]} P{i+1} → {driver['Abbreviation']} ({driver['TeamName']}) - {driver['PodiumProbability']:.0%} confidence")
print("=" * 50)