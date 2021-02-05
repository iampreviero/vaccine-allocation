import datetime as dt
from copy import deepcopy
from typing import List, Dict, Union, Optional
from gurobipy import GurobiError
import pandas as pd
import us

from pipeline.constants import *
from models.mortality_rate_estimator import MortalityRateEstimator


def get_population_by_state_and_risk_class(pop_df: pd.DataFrame) -> np.ndarray:

    states = list(pop_df.state.unique())
    population = np.zeros((len(states), len(RISK_CLASSES)))

    for j, state in enumerate(states):
        for k, risk_class in enumerate(RISK_CLASSES):
            population[j, k] = pop_df[
                (pop_df["min_age"] >= risk_class["min_age"])
                & (pop_df["max_age"] <= risk_class["max_age"])
                & (pop_df["state"] == state)
                ]["population"].sum()
    return population


def calculate_n_timesteps(
        start_date: dt.datetime,
        end_date: dt.datetime
) -> int:
    return int(np.round((end_date - start_date).days / DAYS_PER_TIMESTEP))


def get_policy_response_by_state_and_timestep(
        params_df: pd.DataFrame,
        start_date: dt.datetime,
        end_date: dt.datetime
) -> np.ndarray:
    n_timesteps = calculate_n_timesteps(start_date=start_date, end_date=end_date)
    policy_response = np.zeros((N_REGIONS, n_timesteps))
    t = np.arange(n_timesteps) * DAYS_PER_TIMESTEP
    for j, (state, params) in enumerate(params_df.iterrows()):
        offset = (start_date - params["start_date"]).days * DAYS_PER_TIMESTEP
        lockdown_curve = 2 / np.pi * np.arctan(
            -(t + offset - params["intervention_time"]) * params["intervention_rate"] / 20
        )
        reopening_curve = params["jump_magnitude"] * np.exp(
            -((t + offset - params["jump_time"]) / params["jump_decay"]) ** 2 / 2
        )
        policy_response[j, :] = 1 + lockdown_curve + reopening_curve
    return policy_response


def get_baseline_mortality_rate_estimates(
        cdc_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        start_date: dt.datetime,
        end_date: dt.datetime,
) -> np.ndarray:
    # Compute rescaling factor
    predicted_cases, predicted_deaths = predictions_df[
        (predictions_df["date"] >= start_date)
        & (predictions_df["date"] <= end_date)
        ][["total_detected_cases", "total_detected_deaths"]].max()
    cdc_mortality_rate = (cdc_df["deaths"].sum() / cdc_df["cases"].sum())
    delphi_mortality_rate = predicted_deaths / predicted_cases
    rescaling_factor = delphi_mortality_rate / cdc_mortality_rate if RESCALE_BASELINE else 1.0

    # Compute baseline mortality rates
    baseline_mortality_rate = np.zeros(N_RISK_CLASSES)
    for k, risk_class in enumerate(RISK_CLASSES):
        cases, deaths = cdc_df[
            (cdc_df["min_age"] >= risk_class["min_age"])
            & (cdc_df["max_age"] <= risk_class["max_age"])
            ][["cases", "deaths"]].sum()
        baseline_mortality_rate[k] = deaths / cases * rescaling_factor
    return baseline_mortality_rate


def get_lag_estimates(params_df: pd.DataFrame) -> List[dt.timedelta]:
    lags = np.ceil(MEDIAN_PROGRESSION_TIME + MEDIAN_DETECTION_TIME + (np.log(2) / params_df["death_rate"]))
    lags = np.clip(lags, a_min=MIN_LAG, a_max=MAX_LAG)
    return [dt.timedelta(days=lag) for lag in lags]


def get_mortality_rate_estimates(
        pop_df: pd.DataFrame,
        cdc_df: pd.DataFrame,
        params_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        start_date: dt.datetime,
        end_date: dt.datetime
) -> np.ndarray:
    # Get data
    population = get_population_by_state_and_risk_class(pop_df=pop_df)
    baseline_mortality_rate = get_baseline_mortality_rate_estimates(
        cdc_df=cdc_df,
        predictions_df=predictions_df,
        start_date=start_date,
        end_date=end_date
    )
    lag = get_lag_estimates(params_df=params_df)

    # Initialize grid
    n_timesteps = calculate_n_timesteps(start_date=start_date, end_date=end_date)
    mortality_rate = np.ndarray((N_REGIONS, N_RISK_CLASSES, n_timesteps))

    # Perform estimation for each state
    for j, state in enumerate(pop_df["state"].unique()):
        cases = predictions_df[
            (predictions_df["state"] == state)
            & (predictions_df["date"] >= start_date)
            & (predictions_df["date"] <= end_date)
            ]["total_detected_cases"].diff().dropna().to_numpy()

        deaths = predictions_df[
            (predictions_df["state"] == state)
            & (predictions_df["date"] >= start_date + lag[j])
            & (predictions_df["date"] <= end_date + lag[j])
            ]["total_detected_deaths"].diff().dropna().to_numpy()

        deaths = np.where(deaths / cases <= MAX_MORTALITY_RATE, deaths, MAX_MORTALITY_RATE * cases)
        mortality_rate_estimator = MortalityRateEstimator(
            cases=np.repeat(cases, 1 / DAYS_PER_TIMESTEP) * DAYS_PER_TIMESTEP,
            deaths=np.repeat(deaths, 1 / DAYS_PER_TIMESTEP) * DAYS_PER_TIMESTEP,
            baseline_mortality_rate=baseline_mortality_rate,
            population=population[j, :],
            n_timesteps_per_estimate=N_TIMESTEPS_PER_ESTIMATE,
            max_pct_change=MAX_PCT_CHANGE,
            min_mortality_rate=MIN_MORTALITY_RATE,
            regularization_param=REGULARIZATION_PARAM,
            enforce_monotonicity=True
        )
        try:
            mortality_rate[j, :, :] = mortality_rate_estimator.solve(
                time_limit=TIME_LIMIT,
                feasibility_tol=FEASIBILITY_TOL,
                mip_gap=MIP_GAP,
                output_flag=True,
            )[0]
            print(f"Completed calibration for {state}")
        except GurobiError:
            try:
                print(f"Infeasible problem for {state} - relaxing monotonicity constraints")
                mortality_rate_estimator.enforce_monotonicity = False
                mortality_rate[j, :, :] = mortality_rate_estimator.solve(
                    time_limit=TIME_LIMIT,
                    feasibility_tol=FEASIBILITY_TOL,
                    mip_gap=MIP_GAP,
                    output_flag=True
                )[0]
                print(f"Completed calibration for {state}")
            except GurobiError:
                print(f"Error calibrating mortality rates for {state} - using baseline estimates")
                mortality_rate[j, :, :] = baseline_mortality_rate[:, None]
    return mortality_rate


def get_hospitalization_rate_by_risk_class(cdc_df: pd.DataFrame) -> np.ndarray:
    hospitalization_rate = np.zeros(N_RISK_CLASSES)
    for k, risk_class in enumerate(RISK_CLASSES):
        cases, hospitalizations = cdc_df[
            (cdc_df["min_age"] >= risk_class["min_age"])
            & (cdc_df["max_age"] <= risk_class["max_age"])
            ][["cases", "hospitalizations"]].sum()
        hospitalization_rate[k] = hospitalizations / cases
    return hospitalization_rate


def get_initial_conditions(
        pop_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        start_date: dt.datetime
) -> Dict[str, np.ndarray]:
    # Get population by state and risk class
    population = get_population_by_state_and_risk_class(pop_df=pop_df)

    # Get estimated susceptible, exposed and infectious for start date
    initial_default = np.zeros(population.shape)
    initial_susceptible = deepcopy(initial_default)
    initial_exposed = deepcopy(initial_default)
    initial_infectious = deepcopy(initial_default)
    initial_hospitalized_dying = deepcopy(initial_default)
    initial_hospitalized_recovering = deepcopy(initial_default)
    initial_quarantined_dying = deepcopy(initial_default)
    initial_quarantined_recovering = deepcopy(initial_default)
    initial_undetected_dying = deepcopy(initial_default)
    initial_undetected_recovering = deepcopy(initial_default)
    initial_conditions_df = predictions_df[
        predictions_df["date"] == start_date
        ].sort_values("state")
    for j, (_, state) in enumerate(initial_conditions_df.iterrows()):
        pop_proportions = population[j, :] / population[j, :].sum()
        initial_susceptible[j, :] = state["susceptible"] * pop_proportions
        initial_exposed[j, :] = state["exposed"] * pop_proportions
        initial_infectious[j, :] = state["infectious"] * pop_proportions
        initial_hospitalized_dying[j, :] =  state["hospitalized_dying"] * pop_proportions
        initial_quarantined_dying[j, :] =  state["quarantined_dying"] * pop_proportions
        initial_undetected_dying[j, :] =  state["undetected_dying"] * pop_proportions
        initial_hospitalized_recovering[j, :] =  state["hospitalized_recovering"] * pop_proportions
        initial_quarantined_recovering[j, :] =  state["quarantined_recovering"] * pop_proportions
        initial_undetected_recovering[j, :] =  state["undetected_recovering"] * pop_proportions
    # Return dictionary of all initial conditions
    return dict(
        initial_susceptible=initial_susceptible,
        initial_exposed=initial_exposed,
        initial_infectious=initial_infectious,
        initial_hospitalized_dying=initial_hospitalized_dying,
        initial_hospitalized_recovering=initial_hospitalized_recovering,
        initial_quarantined_dying=initial_quarantined_dying,
        initial_quarantined_recovering=initial_quarantined_recovering,
        initial_undetected_dying=initial_undetected_dying,
        initial_undetected_recovering=initial_undetected_recovering,
        initial_recovered=initial_default,
        population=population
    )


def get_delphi_params(
        pop_df: pd.DataFrame,
        cdc_df: pd.DataFrame,
        params_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        start_date: dt.datetime,
        end_date: dt.datetime,
        mortality_rate_path: Optional[str],
) -> Dict[str, Union[float, np.ndarray]]:
    # Get policy response by state and timestep
    policy_response = get_policy_response_by_state_and_timestep(
        params_df=params_df,
        start_date=start_date,
        end_date=end_date
    )

    # Get mortality rate estimates
    if mortality_rate_path:
        with open(mortality_rate_path, "rb") as fp:
            mortality_rate = np.load(fp)
    else:
        mortality_rate = get_mortality_rate_estimates(
            pop_df=pop_df,
            cdc_df=cdc_df,
            params_df=params_df,
            predictions_df=predictions_df,
            start_date=start_date,
            end_date=end_date
        )

    # Get estimated hospitalization rates from CDC data
    hospitalization_rate = get_hospitalization_rate_by_risk_class(cdc_df=cdc_df)
    hospitalization_rate = hospitalization_rate[None, :, None]

    # Convert median times to rates
    progression_rate = np.log(2) / MEDIAN_PROGRESSION_TIME
    detection_rate = np.log(2) / MEDIAN_DETECTION_TIME
    hospitalized_recovery_rate = np.log(2) / MEDIAN_HOSPITALIZED_RECOVERY_TIME
    unhospitalized_recovery_rate = np.log(2) / MEDIAN_UNHOSPITALIZED_RECOVERY_TIME

    return dict(
        infection_rate=np.array(params_df["infection_rate"]),
        policy_response=policy_response,
        progression_rate=progression_rate,
        detection_rate=detection_rate,
        ihd_transition_rate=detection_rate * DETECTION_PROBABILITY * hospitalization_rate * mortality_rate,
        ihr_transition_rate=detection_rate * DETECTION_PROBABILITY * hospitalization_rate * (1 - mortality_rate),
        iqd_transition_rate=detection_rate * DETECTION_PROBABILITY * (1 - hospitalization_rate) * mortality_rate,
        iqr_transition_rate=detection_rate * DETECTION_PROBABILITY * (1 - hospitalization_rate) * (1 - mortality_rate),
        iud_transition_rate=detection_rate * (1 - DETECTION_PROBABILITY) * mortality_rate,
        iur_transition_rate=detection_rate * (1 - DETECTION_PROBABILITY) * (1 - mortality_rate),
        death_rate = params_df["death_rate"].to_numpy(),
        hospitalized_recovery_rate=hospitalized_recovery_rate,
        unhospitalized_recovery_rate=unhospitalized_recovery_rate,
        mortality_rate=mortality_rate,
        days_per_timestep=DAYS_PER_TIMESTEP
    )


def get_allocation_params(county_pop_df,
                          counties_dists_df,
                          selected_centers_df,
                          baseline_centers_df):

    county_pop_mat = county_pop_df.iloc[:, 2:].to_numpy()
    counties_dists_mat = counties_dists_df.to_numpy()

    valid_pairs = np.argwhere(counties_dists_mat > -1)

    county_to_cities = {idx: np.where(counties_dists_mat[:, idx] > -1)[0]
                        for idx in range(counties_dists_mat.shape[1])}

    county_city_to_distance = {(v[1], v[0]): counties_dists_mat[v[0], v[1]] for v in valid_pairs}

    states = list(county_pop_df.state.unique())
    state_lookup = {idx: state for idx, state in enumerate(states)}

    county_lookup = {idx: fips for idx, fips in enumerate(county_pop_df['county_state'])}
    city_lookup = {idx: name for idx, name in enumerate(selected_centers_df['city'])}

    county_to_state = {idx: states.index(county_pop_df.iloc[idx]['state']) for idx in range(county_pop_df.shape[0])}

    state_to_counties = {idx: list(county_pop_df[county_pop_df.state == state].index)
                         for idx, state in enumerate(states)}

    state_to_cities = {idx: list(selected_centers_df[selected_centers_df.state == state].index)
                       for idx, state in enumerate(states)}

    city_to_state = {idx: states.index(selected_centers_df.iloc[idx]['state'])
                     for idx in range(selected_centers_df.shape[0])}

    cities_budget = 100
    n_cities = selected_centers_df.shape[0]

    # baseline_centers_df.columns = [us.states.lookup(x).name for x in baseline_centers_df.columns]
    # baseline_centers = baseline_centers_df[states].iloc[0, :].to_numpy()

    # This line shows what we want to do:
    # baseline_centers = baseline_centers_df[:,'num_locations'].to_numpy()
    # This line does it in an ugly way, but ensures order is correct no matter how stupid was the person that created the input
    baseline_centers = np.array([baseline_centers_df[baseline_centers_df['state']==s]['num_locations'].item() for i,s in enumerate(states)])

    allocation_params = {
        'population': county_pop_mat,
        'state_lookup': state_lookup,
        'county_lookup': county_lookup,
        'city_lookup': city_lookup,
        'county_to_cities': county_to_cities,
        'county_city_to_distance': county_city_to_distance,
        'county_to_state': county_to_state,
        'state_to_counties': state_to_counties,
        'state_to_cities': state_to_cities,
        'city_to_state': city_to_state,
        'cities_budget': cities_budget,
        'n_cities': n_cities,
        'baseline_centers': baseline_centers
    }

    return allocation_params
