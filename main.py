import fastf1
import numpy as np
import pandas as pd
import matplotlib



# Get the 2023 Azerbaijan Grand Prix Race session
session = fastf1.get_session(2023, 'Azerbaijan', 'Q')

# Load all session data (timing, telemetry, weather, etc.)
session.load()
results = session.results

laps = session.laps

print(laps.columns)