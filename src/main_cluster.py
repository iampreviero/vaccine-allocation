import pandas as pd
import os
import pickle
from copy import deepcopy

from pipeline.scenario import Scenario
from pipeline.constants import *
from datetime import datetime
import sys

if __name__ == "__main__":

    if sys.argv[2] == "exper0":
        from pipeline.grid.exper0 import *
    elif sys.argv[2] == "exper1":
        from pipeline.grid.exper1 import *
    elif sys.argv[2] == "exper2":
        from pipeline.grid.exper2 import *
    elif sys.argv[2] == "exper3":
        from pipeline.grid.exper3 import *

    now = datetime.now()

    dt_string = now.strftime("%Y%m%d-%H%M%S")

    scenario_params_grid = [
        dict(
            **dates,
            vaccine_effectiveness=vaccine_effectiveness,
            daily_vaccine_budget=daily_vaccine_budget
        )
        for dates in DATES_GRID
        for vaccine_effectiveness in VACCINE_EFFECTIVENESS_GRID
        for daily_vaccine_budget in DAILY_VACCINE_BUDGET_GRID
    ]

    baselines_grid = [
        dict(
            baseline=baseline
        )
        for baseline in BASELINES
    ]

    algorithm_params_grid = [
        dict(
            min_allocation_factor=min_allocation_factor,
            political_factor=political_factor,
            balanced_location=balanced_location,
            population_equity_pct=population_equity_pct,
            balanced_distr_locations_pct=balanced_distr_locations_pct,
            initial_solution = initial_solution,
            distance_penalty = distance_penalty
        )
        for min_allocation_factor in MIN_ALLOCATION_FACTOR_GRID
        for political_factor in POLITICAL_FACTOR_GRID
        for balanced_location in BALANCED_LOCATIONS_GRID
        for population_equity_pct in POPULATION_EQUITY_PCT_GRID
        for balanced_distr_locations_pct in BALANCED_DISTR_LOCATIONS_PCT_GRID
        for initial_solution in INITIAL_SOLUTION_GRID
        for distance_penalty in DISTANCE_PENALTY_GRID
    ]

    algorithm_params = algorithm_params_grid[int(sys.argv[1])]

    results_dict_baseline = list()
    results_dict_optimized = list()
    counter_baseline = 0
    counter_optimized = 0

    for i, scenario_params in enumerate(scenario_params_grid):

        start_date = scenario_params["start_date"]
        end_date = scenario_params["start_date"]
        mortality_rate_path = f"{MORTALITY_RATES_PATH}2021-heur.npy"
        reload_mortality_rate = os.path.isfile(mortality_rate_path)

        params_dict = {**scenario_params, **algorithm_params}
        obj_val = Scenario(**params_dict).run(
            model_path=f"{MODEL_PATH_PATH}{int(sys.argv[2])}-{dt_string}-optimized-{int(sys.argv[1])}-{i}.pickle",
            solution_path=f"{OPTIMIZED_SOLUTION_PATH}{int(sys.argv[2])}-{dt_string}-{int(sys.argv[1])}-{i}.pickle",
            mortality_rate_path=mortality_rate_path,
            reload_mortality_rate=reload_mortality_rate
        )
        results_dict_optimized.append(params_dict)
        results_dict_optimized[counter_optimized]["scenario"] = i
        results_dict_optimized[counter_optimized]["optimized"] = int(sys.argv[1])
        results_dict_optimized[counter_optimized]["optimized_obj_val"] = obj_val
        counter_optimized = counter_optimized + 1
        results = pd.DataFrame(results_dict_optimized)
        results.to_csv(f"{RESULTS_PATH}{int(sys.argv[2])}-{dt_string}-optimized-{int(sys.argv[1])}.csv")


        # results["abs_improvement"] = results["baseline_obj_val"] - results["optimized_obj_val"]
        # results["pct_improvement"] = results["abs_improvement"] / results["baseline_obj_val"] * 1e2


    # # FOR DEBUGGING PURPOSES
    # population = pd.read_csv(POPULATION_DATA_PATH)
    # states = population['state'].unique()
    #
    # baseline_solutions = []
    # optimized_solutions = []
    #
    # for i in range(len(scenario_params_grid)):
    #     baseline_solution_path=f"{BASELINE_SOLUTION_PATH}{i}.pickle"
    #     baseline_solutions.append(pickle.load(open(baseline_solution_path, "rb")))
    #     optimized_solution_path=f"{OPTIMIZED_SOLUTION_ATH}{i}.pickle"
    #     optimized_solutions.append(pickle.load(open(optimized_solution_path, "rb")))
    #     V = optimized_solutions[i].vaccinated
    #     print("\n=====")
    #     print("Scenario", i)
    #     print("\nOptimized vs baseline deaths: ", optimized_solutions[i].get_total_deaths(), baseline_solutions[i].get_total_deaths())
    #     print("\nLocations:\n", [(optimized_solutions[i].locations[j], states[j]) for j in range(len(states))])
    #     print("\nVaccines per state:\n", [(V.sum(axis=(1,2))[j], states[j]) for j in range(len(states))])
    #     print("\nVaccines used vs available: ", np.nansum(V), V.shape[2]*6e5)
