import pandas as pd
import os
import pickle

from pipeline.scenario import Scenario
from pipeline.constants import *


if __name__ == "__main__":

    scenario_params_grid = [
        dict(
            **dates,
            vaccine_effectiveness=vaccine_effectiveness,
            daily_vaccine_budget=daily_vaccine_budget,
            min_allocation_factor=min_allocation_factor,
            political_factor=political_factor,
            balanced_location=balanced_location,
            population_equity_pct=population_equity_pct,
            balanced_distr_locations_pct=balanced_distr_locations_pct
        )
        for dates in DATES_GRID
        for vaccine_effectiveness in VACCINE_EFFECTIVENESS_GRID
        for daily_vaccine_budget in DAILY_VACCINE_BUDGET_GRID
        for min_allocation_factor in MIN_ALLOCATION_FACTOR_GRID
        for political_factor in POLITICAL_FACTOR_GRID
        for balanced_location in BALANCED_LOCATIONS_GRID
        for population_equity_pct in POPULATION_EQUITY_PCT_GRID
        for balanced_distr_locations_pct in BALANCED_DISTR_LOCATIONS_PCT_GRID
    ]

    for i, scenario_params in enumerate(scenario_params_grid):

        start_date = scenario_params["start_date"]
        end_date = scenario_params["start_date"]
        mortality_rate_path = f"{MORTALITY_RATES_PATH}2021-heur.npy"
        reload_mortality_rate = os.path.isfile(mortality_rate_path)

        baseline_obj_val, optimized_obj_val = Scenario(**scenario_params).run(
            model_path=f"{MODEL_PATH_PATH}{i}.pickle",
            baseline_solution_path=f"{BASELINE_SOLUTION_PATH}{i}.pickle",
            optimized_solution_path=f"{OPTIMIZED_SOLUTION_ATH}{i}.pickle",
            mortality_rate_path=mortality_rate_path,
            reload_mortality_rate=reload_mortality_rate
        )
        scenario_params_grid[i]["baseline_obj_val"] = baseline_obj_val
        scenario_params_grid[i]["optimized_obj_val"] = optimized_obj_val

        results = pd.DataFrame(scenario_params_grid)
        results["abs_improvement"] = results["baseline_obj_val"] - results["optimized_obj_val"]
        results["pct_improvement"] = results["abs_improvement"] / results["baseline_obj_val"] * 1e2
        results.to_csv(RESULTS_PATH)

    # FOR DEBUGGING PURPOSES
    population = pd.read_csv(POPULATION_DATA_PATH)
    states = population['state'].unique()

    baseline_solutions = []
    optimized_solutions = []

    for i in range(len(scenario_params_grid)):
        baseline_solution_path=f"{BASELINE_SOLUTION_PATH}{i}.pickle"
        baseline_solutions.append(pickle.load(open(baseline_solution_path, "rb")))
        optimized_solution_path=f"{OPTIMIZED_SOLUTION_ATH}{i}.pickle"
        optimized_solutions.append(pickle.load(open(optimized_solution_path, "rb")))
        V = optimized_solutions[i].vaccinated
        print("\n=====")
        print("Scenario", i)
        print("\nOptimized vs baseline deaths: ", optimized_solutions[i].get_total_deaths(), baseline_solutions[i].get_total_deaths())
        print("\nLocations:\n", [(optimized_solutions[i].locations[j], states[j]) for j in range(len(states))])
        print("\nVaccines per state:\n", [(V.sum(axis=(1,2))[j], states[j]) for j in range(len(states))])
        print("\nVaccines used vs available: ", np.nansum(V), V.shape[2]*6e5)
