import numpy as np
import datetime as dt

# Input paths
DELPHI_PARAMS_PATH = "../data/inputs/delphi-parameters.csv"
DELPHI_PREDICTIONS_PATH = "../data/inputs/delphi-predictions.csv"
CDC_DATA_PATH = "../data/inputs/cdc-data.csv"
POPULATION_DATA_PATH = "../data/inputs/population.csv"
COUNTY_POP_DATA_PATH = "../data/inputs/county_pop_clean_fullname.csv"
COUNTY_DISTS_PATH = "../data/inputs/counties_distances_500_fullname.csv"
SELECTED_CENTERS_PATH = "../data/inputs/selected_centers_500_fullname.csv"
BASELINE_ALLOCATION_CITIES_PATH = "../data/inputs/baseline_allocation_cities.csv"
BASELINE_ALLOCATION_POPULATION_PATH = "../data/inputs/baseline_allocation_population.csv"
BASELINE_ALLOCATION_CASES_PATH = "../data/inputs/baseline_allocation_cases.csv"
MORTALITY_RATES_PATH = "../data/inputs/mortality-rates-"


# Output paths
MODEL_PATH_PATH = "../data/outputs/model-"
BASELINE_SOLUTION_PATH = "../data/outputs/baseline-solution-"
OPTIMIZED_SOLUTION_PATH = "../data/outputs/optimized-solution-"
RESULTS_PATH = "../data/outputs/results-"

# Population partition
RISK_CLASSES = [
    dict(min_age=0.0, max_age=9.0),
    dict(min_age=10.0, max_age=49.0),
    dict(min_age=50.0, max_age=59.0),
    dict(min_age=60.0, max_age=69.0),
    dict(min_age=70.0, max_age=79.0),
    dict(min_age=80.0, max_age=np.inf)
]
N_REGIONS = 51  # All 50 US states plus Washington D.C.
N_RISK_CLASSES = len(RISK_CLASSES)

# Time discretization
DAYS_PER_TIMESTEP = 1.0

# Mortality rate estimation parameters
RESCALE_BASELINE = True
N_TIMESTEPS_PER_ESTIMATE = 5
MIN_LAG = 2
MAX_LAG = 20
MAX_PCT_CHANGE = 0.2
MIN_MORTALITY_RATE = 0
MAX_MORTALITY_RATE = 0.3
REGULARIZATION_PARAM = 0.1

# Coordinate descent algorithm parameters
EXPLORATION_TOL = 5e6
TERMINATION_TOL = 5e2
MAX_ITERATIONS = 10
N_EARLY_STOPPING_ITERATIONS = 1
INITIAL_SOLUTION = "cities" # currently supports: cities, population, cases

# Gurobi parameters
TIME_LIMIT = 240
FEASIBILITY_TOL = 1e-3
MIP_GAP = 5e-2
BARRIER_CONV_TOL = 1e-5

# Fixed DELPHI parameters
DETECTION_PROBABILITY = 0.2
MEDIAN_PROGRESSION_TIME = 5.0
MEDIAN_DETECTION_TIME = 2.0
MEDIAN_HOSPITALIZED_RECOVERY_TIME = 10.0
MEDIAN_UNHOSPITALIZED_RECOVERY_TIME = 15.0

# Vaccine and allocation parameters
MAX_ALLOCATION_FACTOR = 10
MIN_ALLOCATION_FACTOR = 0.1
MAX_DECREASE_PCT = 0.1
MAX_INCREASE_PCT = 0.1
MAX_TOTAL_CAPACITY_PCT = None
OPTIMIZE_CAPACITY = False
EXCLUDED_RISK_CLASSES = [0, 5]
POLITICAL_FACTOR = 0
BALANCED_LOCATION = 10
MAX_DISTR_PCT_CHANGE = 0.1
VACCINATION_ENFORCEMENT_WEIGHT = 0.01
POPULATION_EQUITY_PCT = 0.1
BALANCED_DISTR_LOCATIONS_PCT = 0.1
DISTANCE_PENALTY = 1e-4

RUN_BASELINES = False
