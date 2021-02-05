from typing import Dict, Union, List, Tuple, Optional
import pickle

import pandas as pd

from pipeline.constants import *
from pipeline.data_loading import load_and_clean_delphi_predictions, load_and_clean_delphi_params
from pipeline.data_processing import (calculate_n_timesteps,
                                      get_initial_conditions,
                                      get_delphi_params,
                                      get_allocation_params)
from models.prescriptive_delphi_model import PrescriptiveDELPHIModel
import numpy as np

class Scenario:

    def __init__(
            self,
            # Scenario parameters
            start_date: dt.datetime,
            end_date: dt.datetime,
            vaccine_effectiveness: float,
            daily_vaccine_budget: float,
            # Whether to run baseline or optimization
            baseline: str = "none",
            # Optimization algorithm parameters
            min_allocation_factor: float = MIN_ALLOCATION_FACTOR,
            max_allocation_factor: float = MAX_ALLOCATION_FACTOR,
            max_increase_pct: float = MAX_INCREASE_PCT,
            max_decrease_pct: float = MAX_DECREASE_PCT,
            political_factor: float = POLITICAL_FACTOR,
            balanced_location: float = BALANCED_LOCATION,
            excluded_risk_classes: List[int] = EXCLUDED_RISK_CLASSES,
            max_total_capacity: Optional[float] = None,
            optimize_capacity: bool = OPTIMIZE_CAPACITY,
            max_distr_pct_change: float = MAX_DISTR_PCT_CHANGE,
            population_equity_pct: float = POPULATION_EQUITY_PCT,
            balanced_distr_locations_pct: float = BALANCED_DISTR_LOCATIONS_PCT,
            vaccination_enforcement_weight: float = VACCINATION_ENFORCEMENT_WEIGHT,
            distance_penalty: float = DISTANCE_PENALTY,
            initial_solution: str = "cities"
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.vaccine_effectiveness = vaccine_effectiveness
        self.daily_vaccine_budget = daily_vaccine_budget
        self.baseline = baseline
        self.max_total_capacity = max_total_capacity if max_total_capacity else daily_vaccine_budget
        self.max_allocation_factor = max_allocation_factor
        self.min_allocation_factor = min_allocation_factor
        self.max_increase_pct = max_increase_pct
        self.max_decrease_pct = max_decrease_pct
        self.excluded_risk_classes = excluded_risk_classes
        self.optimize_capacity = optimize_capacity
        self.political_factor = political_factor
        self.balanced_location = balanced_location
        self.max_distr_pct_change = max_distr_pct_change
        self.population_equity_pct = population_equity_pct
        self.vaccination_enforcement_weight = vaccination_enforcement_weight
        self.balanced_distr_locations_pct = balanced_distr_locations_pct
        self.initial_solution = initial_solution
        self.distance_penalty = distance_penalty

    def get_vaccine_params(
            self,
            total_pop: float,
    ) -> Dict[str, Union[float, np.ndarray]]:
        n_timesteps = calculate_n_timesteps(start_date=self.start_date, end_date=self.end_date)
        return dict(
            vaccine_effectiveness=self.vaccine_effectiveness,
            vaccine_budget=np.ones(n_timesteps) * self.daily_vaccine_budget,
            max_total_capacity=self.max_total_capacity,
            max_allocation_pct=self.daily_vaccine_budget / total_pop * self.max_allocation_factor,
            min_allocation_pct=self.daily_vaccine_budget / total_pop * self.min_allocation_factor,
            max_decrease_pct=self.max_increase_pct,
            max_increase_pct=self.max_decrease_pct,
            excluded_risk_classes=np.array(self.excluded_risk_classes) if self.excluded_risk_classes else np.array([]).astype(int),
            optimize_capacity=self.optimize_capacity,
            max_distr_pct_change=self.max_distr_pct_change
        )

    def load_model(
            self,
            mortality_rate_path: Optional[str] = None
    ) -> PrescriptiveDELPHIModel:

        # Load raw data
        params_df = load_and_clean_delphi_params(DELPHI_PARAMS_PATH)
        predictions_df = load_and_clean_delphi_predictions(DELPHI_PREDICTIONS_PATH)
        cdc_df = pd.read_csv(CDC_DATA_PATH)
        pop_df = pd.read_csv(POPULATION_DATA_PATH)
        county_pop_df = pd.read_csv(COUNTY_POP_DATA_PATH)
        counties_dists_df = pd.read_csv(COUNTY_DISTS_PATH, index_col=0)
        selected_centers_df = pd.read_csv(SELECTED_CENTERS_PATH)
        if self.baseline == "cities":
            baseline_centers_df = pd.read_csv(BASELINE_ALLOCATION_CITIES_PATH)
        elif self.baseline == "population":
            baseline_centers_df = pd.read_csv(BASELINE_ALLOCATION_POPULATION_PATH)
        elif self.baseline == "cases":
            baseline_centers_df = pd.read_csv(BASELINE_ALLOCATION_CASES_PATH)
        else:
            if self.initial_solution == "cities":
                baseline_centers_df = pd.read_csv(BASELINE_ALLOCATION_CITIES_PATH)
            elif self.initial_solution == "population":
                baseline_centers_df = pd.read_csv(BASELINE_ALLOCATION_POPULATION_PATH)
            elif self.initial_solution == "cases":
                baseline_centers_df = pd.read_csv(BASELINE_ALLOCATION_CASES_PATH)

        # Get processed data for model
        initial_conditions = get_initial_conditions(
            pop_df=pop_df,
            predictions_df=predictions_df,
            start_date=self.start_date
        )
        delphi_params = get_delphi_params(
            pop_df=pop_df,
            cdc_df=cdc_df,
            params_df=params_df,
            predictions_df=predictions_df,
            start_date=self.start_date,
            end_date=self.end_date,
            mortality_rate_path=mortality_rate_path
        )
        vaccine_params = self.get_vaccine_params(total_pop=initial_conditions["population"].sum())

        allocation_params = get_allocation_params(county_pop_df=county_pop_df,
                                                  counties_dists_df=counties_dists_df,
                                                  selected_centers_df=selected_centers_df,
                                                  baseline_centers_df=baseline_centers_df)
        allocation_params["political_factor"] = self.political_factor
        allocation_params["balanced_location"] = self.balanced_location
        allocation_params["population_equity_pct"] = self.population_equity_pct
        allocation_params["vaccination_enforcement_weight"] = self.vaccination_enforcement_weight
        allocation_params["balanced_distr_locations_pct"] = self. balanced_distr_locations_pct
        allocation_params["distance_penalty"] = self.distance_penalty
        # Return prescriptive DELPHI model object
        return PrescriptiveDELPHIModel(
            initial_conditions=initial_conditions,
            delphi_params=delphi_params,
            vaccine_params=vaccine_params,
            allocation_params=allocation_params
        )

    def run(
            self,
            mortality_rate_path: str,
            model_path: Optional[str] = None,
            solution_path: Optional[str] = None,
            reload_mortality_rate: bool = False
    ) -> Tuple[float, float]:

        print("Loading model...")
        model = self.load_model(mortality_rate_path=mortality_rate_path if reload_mortality_rate else None)
        if not reload_mortality_rate:
            with open(mortality_rate_path, "wb") as fp:
                np.save(fp, model.mortality_rate)

        if not RUN_BASELINES:
            print("Optimizing...")
            solution = model.optimize(
                exploration_tol=EXPLORATION_TOL,
                termination_tol=TERMINATION_TOL,
                max_iterations=MAX_ITERATIONS,
                n_early_stopping_iterations=N_EARLY_STOPPING_ITERATIONS,
                log=True
            )
        else:
            print("Running baseline...")
            solution = model.simulate(prioritize_allocation=False, initial_solution_allocation=True)

        if solution_path:
            with open(solution_path, "wb") as fp:
                pickle.dump(solution, fp)
        if model_path:
            with open(model_path, "wb") as fp:
                pickle.dump(model, fp)

        return solution.get_objective_value()
