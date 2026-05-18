import pandas as pd


races_df = pd.read_csv('/Users/rishitsingh/Desktop/F1_podium/data/race_results.csv')
quali_df = pd.read_csv('/Users/rishitsingh/Desktop/F1_podium/data/quali_results.csv')

#Getting a dtaframe that can be put in the model for training and testing

model_df = pd.merge(quali_df, races_df, on=['DriverNumber', 'Abbreviation', 'TeamName', 'Year', 'Round', 'EventName'], suffixes=('_quali', '_race'))
model_df = model_df.sort_values(['Year', 'Round'])

# we need to to add some columns to our model_df
model_df['DriverAvgLast3'] = model_df.groupby('DriverNumber')['Position_race'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())

# Fill NaN (first race of each driver) with overall average
model_df['DriverAvgLast3'] = model_df['DriverAvgLast3'].fillna(model_df['Position_race'].mean())

#track History of driver at the track
model_df['TrackHistoryAvg'] = model_df.groupby(['DriverNumber', 'EventName'])['Position_race'].transform(lambda x: x.shift(1).expanding().mean())

# Fill NaN (first race of each driver at the track) with overall average
model_df['TrackHistoryAvg'] = model_df['TrackHistoryAvg'].fillna(model_df['Position_race'].mean())

# Team Average Grid Position this season
model_df['TeamAvgSeason'] = model_df.groupby(['TeamName', 'Year'])['Position_race'].transform(lambda x: x.shift(1).expanding().mean())

# Fill NaN
model_df['TeamAvgSeason'] = model_df['TeamAvgSeason'].fillna(model_df['Position_race'].mean())

# total points this season before the race
model_df['CumPoints'] = model_df.groupby(['DriverNumber', 'Year'])['Points_race'].transform(lambda x: x.shift(1).expanding().sum())

# ChampionShip Position for the driver
model_df['ChampionshipPos'] = model_df.groupby(['Year', 'Round'])['CumPoints'].rank(ascending=False, method='min')

# Round 1 everyone has 0 points so CumPoints = NaN
# Fill with mean place (10) since no points = middle of the grid
model_df['ChampionshipPos'] = model_df['ChampionshipPos'].fillna(10)

# Processed Dataframe 
Processed_df = model_df[['DriverNumber', 'Abbreviation', 'TeamName', 'Year', 'Round', 'EventName', 'GridPosition_race', 'Position_race', 'Points_race', 'DriverAvgLast3', 'TrackHistoryAvg', 'TeamAvgSeason', 'ChampionshipPos', 'Podium']]

# save as csv
Processed_df.to_csv('/Users/rishitsingh/Desktop/F1_podium/data/processed_data.csv', index=False)



