import numpy as np
import datetime as dt

# Scenarios
DATES_GRID = [
    # dict(start_date=dt.datetime(2020, 4, 20), end_date=dt.datetime(2020, 7, 15)),
    dict(start_date=dt.datetime(2021, 2, 1), end_date=dt.datetime(2021, 4, 30))
]
VACCINE_EFFECTIVENESS_GRID = [0.9]
DAILY_VACCINE_BUDGET_GRID = [0.5e6, 1e6, 1.5e6, 2e6]

# Baselines
BASELINES = ["cases","population"]
RUN_BASELINES = False

# Algorithm parameters
MIN_ALLOCATION_FACTOR_GRID = [0.1]
POLITICAL_FACTOR_GRID = [0]
BALANCED_LOCATIONS_GRID = [5]
POPULATION_EQUITY_PCT_GRID = [0.1]
BALANCED_DISTR_LOCATIONS_PCT_GRID = [1]
INITIAL_SOLUTION_GRID = ["cases"]
DISTANCE_PENALTY_GRID = [0, 1e-6, 1e-5, 1e-4, 1e-3]
