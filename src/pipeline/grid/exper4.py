import numpy as np
import datetime as dt

# Scenarios
DATES_GRID = [
    # dict(start_date=dt.datetime(2020, 4, 20), end_date=dt.datetime(2020, 7, 15)),
    dict(start_date=dt.datetime(2021, 2, 1), end_date=dt.datetime(2021, 4, 30))
]
VACCINE_EFFECTIVENESS_GRID = [0.9]
DAILY_VACCINE_BUDGET_GRID = [1e6]
VACCINATED_INFECTION_GRID = [True]
CDC_INFECTION_RATE_GRID = [False, True]

# Baselines
BASELINES = ["cases","population"]

# Algorithm parameters
MIN_ALLOCATION_FACTOR_GRID = [0.1]
POLITICAL_FACTOR_GRID = [0]
BALANCED_LOCATIONS_GRID = [5]
POPULATION_EQUITY_PCT_GRID = [0.1]
BALANCED_DISTR_LOCATIONS_PCT_GRID = [0.5]
INITIAL_SOLUTION_GRID = ["cities"]
DISTANCE_PENALTY_GRID = [1e-6]
LOCATIONS_PER_STATE_FIXED_GRID = [False, True]
RANDOM_INFECTION_RATE_GRID = [False]
RANDOM_MORTALITY_RATE_GRID = [False]
