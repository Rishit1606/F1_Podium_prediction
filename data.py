import os
import fastf1
import pandas as pd
import time


os.makedirs('/Users/rishitsingh/Desktop/F1_podium/cache', exist_ok=True)
os.makedirs('/Users/rishitsingh/Desktop/F1_podium/data',  exist_ok=True)


# Enable caching to speed up data retrieval 
fastf1.Cache.enable_cache('/Users/rishitsingh/Desktop/F1_podium/cache')

# We will drop each race's dataframe into this list inside the loop
all_quali_results = []
all_quali_laps    = []
all_race_results  = []
all_race_laps     = []

for year in range(2018, 2027):
    # Get the event schedule for the specified year
    schedule = fastf1.get_event_schedule(year, include_testing=False)  # Set include_testing to False to exclude pre-season testing sessions

    schedule_df = pd.DataFrame(schedule)          # ← convert schedule to DataFrame
    
    # Counts unique values in a column(RoundNumber)
    Races =len(schedule_df['RoundNumber'])
    print(Races)

    Race_Number = 1
    
    while Race_Number <= Races:

        # ===================================================
        #             Qualifying Session
        # ===================================================

        try:
        
            # Get the qualifying session for the current race
            session = fastf1.get_session(year, Race_Number, 'Q')
            session.load(laps=True, telemetry=False, weather=False, messages=False)

            # Get the results of the qualifying session, These also tells us every stat about the session like lap times, driver names, team names, etc.
            results = session.results
            laps = session.laps

            results_Q_df = pd.DataFrame(results)          # ← convert results to DataFrame
            results_Q_df = results_Q_df[['DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Points']]

            # Without Year and Round we won't know which race this row belongs to after combining all races
            results_Q_df['Year']  = year
            results_Q_df['Round'] = Race_Number

            
            laps_Q_df = laps[['DriverNumber', 'LapTime', 'LapNumber', 'IsPersonalBest']]
            results_Q_df['EventName'] = session.event['EventName']

            # Without Year and Round we won't know which race this lap belongs to after combining all races
            laps_Q_df['Year']  = year
            laps_Q_df['Round'] = Race_Number

            # Append the DataFrames to the list
            all_quali_results.append(results_Q_df)
            all_quali_laps.append(laps_Q_df)

        except Exception as e:
            print(f"Error processing Qualifying session for Year: {year}, Round: {Race_Number}. Error: {e}")
            




        # ===================================================
        #             Race Session
        # ===================================================


        try:

            # Get the race session for the current race
            session = fastf1.get_session(year, Race_Number, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)

            # Get the results of the race session, These also tells us every stat about the session like lap times, driver names, team names, etc.
            results = session.results
            laps = session.laps

            results_R_df = pd.DataFrame(results)          # ← convert results to DataFrame
            results_R_df = results_R_df[['DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Points']]
            results_R_df['EventName'] = session.event['EventName']

            results_R_df['Year']  = year
            results_R_df['Round'] = Race_Number

            #make podium winners 1 and rest 0
            results_R_df['Podium'] = results_R_df['Position'].apply(lambda x: 1 if pd.notna(x) and float(x) in [1.0, 2.0, 3.0] else 0)

            
            laps_R_df = laps[['DriverNumber', 'LapTime', 'LapNumber', 'IsPersonalBest']]

            laps_R_df['Year']  = year
            laps_R_df['Round'] = Race_Number

            # Append the DataFrames to the list
            all_race_results.append(results_R_df)
            all_race_laps.append(laps_R_df)


        except Exception as e:
            print(f"Error processing Race session for Year: {year}, Round: {Race_Number}. Error: {e}")
            

        Race_Number+=1
        time.sleep(1)  # Sleep for 1 second to avoid overwhelming the API

pd.concat(all_quali_results, ignore_index=True).to_csv('data/quali_results.csv', index=False)
# pd.concat(all_quali_laps,    ignore_index=True).to_csv('data/quali_laps.csv',    index=False)
pd.concat(all_race_results,  ignore_index=True).to_csv('data/race_results.csv',  index=False)
# pd.concat(all_race_laps,     ignore_index=True).to_csv('data/race_laps.csv',     index=False)

