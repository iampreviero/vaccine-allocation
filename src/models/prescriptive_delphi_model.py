from typing import Tuple, Union, Optional, Dict

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from gurobipy import GurobiError


class DELPHISolution:

    def __init__(
            self,
            susceptible: np.ndarray,
            exposed: np.ndarray,
            infectious: np.ndarray,
            hospitalized_dying: np.ndarray,
            hospitalized_recovering: np.ndarray,
            quarantined_dying: np.ndarray,
            quarantined_recovering: np.ndarray,
            undetected_dying: np.ndarray,
            undetected_recovering: np.ndarray,
            deceased: np.ndarray,
            recovered: np.ndarray,
            eligible: np.ndarray,
            vaccinated: np.ndarray,
            locations: np.ndarray,
            location_indicator: np.ndarray,
            vaccine_distribution: np.ndarray,
            days_per_timestep: float = 1.0,
    ):
        """
        Instantiate a container object for a DELPHI solution.

        :param susceptible: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        susceptible population by region and risk class at each timestep
        :param exposed: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the exposed
        population by region and risk class at each timestep
        :param infectious: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        infectious population by region and risk class at each timestep
        :param hospitalized_dying: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents
        the hospitalized dying population by region and risk class at each timestep
        :param hospitalized_recovering: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that
        represents the hospitalized recovering population by region and risk class at each timestep
        :param quarantined_dying: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents
        the quarantined dying population by region and risk class at each timestep
        :param quarantined_recovering: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that
        represents the quarantined recovering population by region and risk class at each timestep
        :param undetected_dying: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        undetected dying population by region and risk class at each timestep
        :param undetected_recovering: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that
        represents the undetected recovering population by region and risk class at each timestep
        :param recovered: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        recovered population by region and risk class at each timestep
        :param deceased: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        deceased population by region and risk class at each timestep
        :param eligible: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        population eligible for vaccination by region and risk class at each timestep
        :param vaccinated: a numpy array of shape (n_regions, n_risk_classes, n_timesteps + 1) that represents the
        number vaccines allocated by region and risk class at each timestep
        :param days_per_timestep: a positive float that specifies the number of days per timestep used in the
        discretization scheme of the solution
        """

        # Initialize functional states
        self.susceptible = susceptible
        self.exposed = exposed
        self.infectious = infectious
        self.hospitalized_dying = hospitalized_dying
        self.hospitalized_recovering = hospitalized_recovering
        self.hospitalized = hospitalized_dying + hospitalized_recovering
        self.quarantined_dying = quarantined_dying
        self.quarantined_recovering = quarantined_recovering
        self.quarantined = quarantined_dying + quarantined_recovering
        self.undetected_dying = undetected_dying
        self.undetected_recovering = undetected_recovering
        self.undetected = undetected_dying + undetected_recovering
        self.deceased = deceased
        self.recovered = recovered
        self.eligible = eligible
        self.vaccinated = vaccinated
        self.locations = locations
        self.days_per_timestep = days_per_timestep
        self.location_indicator = location_indicator
        self.vaccine_distribution = vaccine_distribution

    def get_total_deaths(self) -> float:
        return self.deceased[:, :, -1].sum()

    def get_objective_value(self) -> float:
        # (self.deceased + self.hospitalized_dying + self.quarantined_dying + self.undetected_dying)[:, :,
        # -2].sum()

        total = (self.deceased + self.hospitalized_dying + self.quarantined_dying)[:, :, -2].sum()
        return total

    def get_total_cases(self) -> float:
        infectious = self.infectious.sum(axis=(0, 1))
        return 0.5 * (infectious[1:] + infectious[:-1]).sum() * self.days_per_timestep


class PrescriptiveDELPHIModel:

    def __init__(
            self,
            initial_conditions: Dict[str, np.ndarray],
            delphi_params: Dict[str, Union[float, np.ndarray]],
            vaccine_params: Dict[str, Union[float, bool, np.ndarray]],
            allocation_params: Dict[str, np.ndarray],
            n_simulate_timesteps_per_optimize_step: int = 7
    ):
        """
        Instantiate a prescriptive DELPHI model with initial conditions and parameter estimates.

        :param initial_conditions: a dictionary containing initial condition arrays
        :param delphi_params: a dictionary containing the estimated DELPHI model parameters
        :param vaccine_params: a dictionary containing vaccine-related parameters
        """

        # Set initial conditions
        self.initial_susceptible = initial_conditions["initial_susceptible"]
        self.initial_exposed = initial_conditions["initial_exposed"]
        self.initial_infectious = initial_conditions["initial_infectious"]
        self.initial_hospitalized_dying = initial_conditions["initial_hospitalized_dying"]
        self.initial_hospitalized_recovering = initial_conditions["initial_hospitalized_recovering"]
        self.initial_quarantined_dying = initial_conditions["initial_quarantined_dying"]
        self.initial_quarantined_recovering = initial_conditions["initial_quarantined_recovering"]
        self.initial_undetected_dying = initial_conditions["initial_undetected_dying"]
        self.initial_undetected_recovering = initial_conditions["initial_undetected_recovering"]
        self.initial_recovered = initial_conditions["initial_recovered"]

        # Set population demographics of shape [n_counties / n_state, n_risk_classes]
        self.state_population = initial_conditions["population"]
        self.county_population = allocation_params["population"]

        # Set DELPHI model parameters
        self.infection_rate = delphi_params["infection_rate"]
        self.policy_response = delphi_params["policy_response"]
        self.progression_rate = delphi_params["progression_rate"]
        self.detection_rate = delphi_params["detection_rate"]
        self.ihd_transition_rate = delphi_params["ihd_transition_rate"]
        self.ihr_transition_rate = delphi_params["ihr_transition_rate"]
        self.iqd_transition_rate = delphi_params["iqd_transition_rate"]
        self.iqr_transition_rate = delphi_params["iqr_transition_rate"]
        self.iud_transition_rate = delphi_params["iud_transition_rate"]
        self.iur_transition_rate = delphi_params["iur_transition_rate"]
        self.death_rate = delphi_params["death_rate"]
        self.hospitalized_recovery_rate = delphi_params["hospitalized_recovery_rate"]
        self.unhospitalized_recovery_rate = delphi_params["unhospitalized_recovery_rate"]
        self.mortality_rate = delphi_params["mortality_rate"]
        self.days_per_timestep = delphi_params["days_per_timestep"]

        # Set vaccine parameters
        self.vaccine_effectiveness = vaccine_params["vaccine_effectiveness"]
        self.vaccine_budget = vaccine_params["vaccine_budget"]
        self.max_total_capacity = vaccine_params["max_total_capacity"]
        self.max_allocation_pct = vaccine_params["max_allocation_pct"]
        self.min_allocation_pct = vaccine_params["min_allocation_pct"]
        self.max_decrease_pct = vaccine_params["max_decrease_pct"]
        self.max_increase_pct = vaccine_params["max_increase_pct"]
        self.excluded_risk_classes = vaccine_params["excluded_risk_classes"]
        self.max_distr_pct_change = vaccine_params["max_distr_pct_change"]

        # Set allocation parameters
        self.county_to_cities = allocation_params["county_to_cities"]
        self.county_city_to_distance = allocation_params["county_city_to_distance"]
        self.county_to_state = allocation_params["county_to_state"]
        self.state_to_counties = allocation_params["state_to_counties"]
        self.state_to_cities = allocation_params["state_to_cities"]
        self.city_to_state = allocation_params["city_to_state"]
        self.distance_penalty = allocation_params["distance_penalty"]
        self.cities_budget = allocation_params["cities_budget"]
        self.baseline_centers = allocation_params["baseline_centers"]
        self.population_equity_pct = allocation_params["population_equity_pct"]
        self.vaccination_enforcement_weight = allocation_params["vaccination_enforcement_weight"]
        self.balanced_distr_locations_pct = allocation_params["balanced_distr_locations_pct"]

        # Initialize helper attributes

        # Locations & risk classes dimensions
        self.n_regions = self.initial_susceptible.shape[0]
        self.n_counties = self.county_population.shape[0]
        self.n_risk_classes = self.initial_susceptible.shape[1]
        self.n_cities = allocation_params["n_cities"]
        self.n_included_risk_classes = self.n_risk_classes - self.excluded_risk_classes.shape[0]
        self.regions = np.arange(self.n_regions)
        self.risk_classes = np.arange(self.n_risk_classes)
        self.included_risk_classes = np.array([k for k in self.risk_classes if k not in self.excluded_risk_classes])

        # Time dimensions
        self.n_simulate_timesteps_per_optimize_step = n_simulate_timesteps_per_optimize_step
        self.n_timesteps = self.vaccine_budget.shape[0]
        self.timesteps = np.arange(self.n_timesteps)
        self.political_factor = allocation_params["political_factor"]
        self.balanced_location = allocation_params["balanced_location"]
        self.optimize_timesteps = np.array(
            [x for x in self.timesteps if x % self.n_simulate_timesteps_per_optimize_step == 0])
        self.n_optimize_timesteps = len(self.optimize_timesteps)
        self.optimize_timestep_to_timestep = {}
        self.timestep_to_optimize_timestep = {}
        for opt_timestep in self.optimize_timesteps:
            self.optimize_timestep_to_timestep[opt_timestep] = np.arange(start=opt_timestep, stop=min(
                opt_timestep + self.n_simulate_timesteps_per_optimize_step, self.n_timesteps))

        # Storage attributes
        self.trajectories = []
        self.solutions = []

    def simulate(
            self,
            vaccinated: Optional[np.ndarray] = None,
            locations: Optional[np.ndarray] = None,
            location_indicator: Optional[np.ndarray] = None,
            vaccine_distribution: Optional[np.ndarray] = None,
            randomize_allocation: bool = False,
            prioritize_allocation: bool = False,
            initial_solution_allocation: bool = False
    ) -> DELPHISolution:
        """
        Solve DELPHI IVP using a forward difference scheme.

        :param vaccinated: a numpy array of (n_regions, n_classes, n_timesteps + 1) that represents a feasible
        allocation of vaccines by region and risk class at each timestep
        :param locations:
        :param randomize_allocation: a boolean that specifies whether to randomly generate a feasible  allocation policy
        based on an exponential distribution if none provided (default False)
        :param prioritize_allocation: a boolean that specifies whether to prioritize allocation to high risk
        individuals within each region if no allocation policy is provided (default False)
        :param initial_solution_allocation:
        :return: a DELPHISolution object
        """

        # Initialize functional states
        dims = (self.n_regions, self.n_risk_classes, self.n_timesteps + 1)
        susceptible = np.zeros(dims)
        exposed = np.zeros(dims)
        infectious = np.zeros(dims)
        hospitalized_dying = np.zeros(dims)
        hospitalized_recovering = np.zeros(dims)
        quarantined_dying = np.zeros(dims)
        quarantined_recovering = np.zeros(dims)
        undetected_dying = np.zeros(dims)
        undetected_recovering = np.zeros(dims)
        deceased = np.zeros(dims)
        recovered = np.zeros(dims)
        eligible = np.zeros(dims)

        # Initialize control variable if none provided
        allocate_vaccines = vaccinated is None
        if allocate_vaccines:
            vaccinated = np.zeros(dims)

        # Set initial conditions
        susceptible[:, :, 0] = self.initial_susceptible
        exposed[:, :, 0] = self.initial_exposed
        infectious[:, :, 0] = self.initial_infectious
        hospitalized_dying[:, :, 0] = self.initial_hospitalized_dying
        hospitalized_recovering[:, :, 0] = self.initial_hospitalized_recovering
        quarantined_dying[:, :, 0] = self.initial_quarantined_dying
        quarantined_recovering[:, :, 0] = self.initial_quarantined_recovering
        undetected_dying[:, :, 0] = self.initial_undetected_dying
        undetected_recovering[:, :, 0] = self.initial_undetected_recovering
        recovered[:, :, 0] = self.initial_recovered
        eligible[:, :, 0] = self.initial_susceptible

        # Generate risk classes and region ranks
        risk_class_priority_rankings = np.argsort(-self.mortality_rate.mean(axis=(0, 2)))
        region_priority_rankings = np.random.permutation(self.regions)  # only used for randomized allocation

        # Propagate discrete DELPHI dynamics with vaccine allocation heuristic
        for t in self.timesteps:

            # Allocate vaccines if required
            if eligible.sum() and allocate_vaccines:

                if initial_solution_allocation:
                    locations = self.baseline_centers
                    regional_budget = self.vaccine_budget[t]/self.cities_budget * self.baseline_centers
                    for k in risk_class_priority_rankings:
                        if k in self.included_risk_classes:
                            vaccinated[:, k, t] = np.minimum(regional_budget, eligible[:, k, t])
                            regional_budget -= vaccinated[:, k, t]
                            if regional_budget.sum() == 0:
                                break

                else:
                    # If random allocation specified, generate feasible allocation
                    if randomize_allocation:
                        min_vaccinated = self.min_allocation_pct * eligible[:, :, t]
                        additional_budget = self.vaccine_budget[t] - min_vaccinated.sum()
                        for j in region_priority_rankings:
                            regional_budget = np.minimum(
                                additional_budget,
                                self.max_allocation_pct * self.state_population[j, :].sum() - min_vaccinated[j, :].sum()
                            )
                            additional_budget -= regional_budget
                            for k in risk_class_priority_rankings:
                                if k in self.included_risk_classes:
                                    vaccinated[j, k, t] = np.minimum(regional_budget, eligible[j, k, t]) \
                                                          + min_vaccinated[j, k]
                                    regional_budget -= vaccinated[j, k, t]
                                    if regional_budget <= 0:
                                        break
                            if additional_budget <= 0:
                                break

                    # Else use baseline policy that orders region-wise allocation by risk class
                    else:
                        if prioritize_allocation:
                            regional_budget = np.minimum(
                                eligible[:, :, t].sum(axis=1) / eligible[:, :, t].sum() * self.vaccine_budget[t],
                                self.max_allocation_pct * self.state_population.sum(axis=1)
                            )
                            for k in risk_class_priority_rankings:
                                if k in self.included_risk_classes:
                                    vaccinated[:, k, t] = np.minimum(regional_budget, eligible[:, k, t])
                                    regional_budget -= vaccinated[:, k, t]
                                    if regional_budget.sum() == 0:
                                        break
                        else:
                            for k in self.included_risk_classes:
                                vaccinated[:, k, t] = self.vaccine_budget[t] * self.state_population[:, k] / \
                                                      self.state_population[:, self.included_risk_classes].sum()

            # Else ensure that the allocated vaccines do not exceed the eligible population
            vaccinated[:, :, t] = np.minimum(vaccinated[:, :, t], eligible[:, :, t])

            # Apply Euler forward difference scheme with clipping of negative values
            for j in self.regions:
                susceptible[j, :, t + 1] = susceptible[j, :, t] - self.vaccine_effectiveness * vaccinated[j, :, t] - (
                        self.infection_rate[j] * self.policy_response[j, t] / self.state_population[j, :].sum()
                        * (susceptible[j, :, t] - self.vaccine_effectiveness * vaccinated[j, :, t])
                        * infectious[j, :, t].sum()
                ) * self.days_per_timestep
                eligible[j, :, t + 1] = eligible[j, :, t] - vaccinated[j, :, t] - (
                        self.infection_rate[j] * self.policy_response[j, t] / self.state_population[j, :].sum()
                        * (susceptible[j, :, t] - self.vaccine_effectiveness * vaccinated[j, :, t])
                        * infectious[j, :, t].sum()
                ) * self.days_per_timestep
                exposed[j, :, t + 1] = exposed[j, :, t] + (
                        self.infection_rate[j] * self.policy_response[j, t] / self.state_population[j, :].sum()
                        * (susceptible[j, :, t] - self.vaccine_effectiveness * vaccinated[j, :, t]) *
                        infectious[j, :, t].sum()
                        - self.progression_rate * exposed[j, :, t]
                ) * self.days_per_timestep
            susceptible[:, :, t + 1] = np.maximum(susceptible[:, :, t + 1], 0)
            exposed[:, :, t + 1] = np.maximum(exposed[:, :, t + 1], 0)
            eligible[:, :, t + 1] = np.maximum(eligible[:, :, t + 1], 0)

            infectious[:, :, t + 1] = infectious[:, :, t] + (
                    self.progression_rate * exposed[:, :, t]
                    - self.detection_rate * infectious[:, :, t]
            ) * self.days_per_timestep
            infectious[:, :, t + 1] = np.maximum(infectious[:, :, t + 1], 0)

            hospitalized_dying[:, :, t + 1] = hospitalized_dying[:, :, t] + (
                    self.ihd_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.death_rate[:, None] * hospitalized_dying[:, :, t]
            ) * self.days_per_timestep
            hospitalized_dying[:, :, t + 1] = np.maximum(hospitalized_dying[:, :, t + 1], 0)

            hospitalized_recovering[:, :, t + 1] = hospitalized_recovering[:, :, t] + (
                    self.ihr_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.hospitalized_recovery_rate * hospitalized_recovering[:, :, t]
            ) * self.days_per_timestep
            hospitalized_recovering[:, :, t + 1] = np.maximum(hospitalized_recovering[:, :, t + 1], 0)

            quarantined_dying[:, :, t + 1] = quarantined_dying[:, :, t] + (
                    self.iqd_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.death_rate[:, None] * quarantined_dying[:, :, t]
            ) * self.days_per_timestep
            quarantined_dying[:, :, t + 1] = np.maximum(quarantined_dying[:, :, t + 1], 0)

            quarantined_recovering[:, :, t + 1] = quarantined_recovering[:, :, t] + (
                    self.iqr_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.unhospitalized_recovery_rate * quarantined_recovering[:, :, t]
            ) * self.days_per_timestep
            quarantined_recovering[:, :, t + 1] = np.maximum(quarantined_recovering[:, :, t + 1], 0)

            undetected_dying[:, :, t + 1] = undetected_dying[:, :, t] + (
                    self.iud_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.death_rate[:, None] * undetected_dying[:, :, t]
            ) * self.days_per_timestep
            undetected_dying[:, :, t + 1] = np.maximum(undetected_dying[:, :, t + 1], 0)

            undetected_recovering[:, :, t + 1] = undetected_recovering[:, :, t] + (
                    self.iur_transition_rate[:, :, t] * infectious[:, :, t]
                    - self.unhospitalized_recovery_rate * undetected_recovering[:, :, t]
            ) * self.days_per_timestep
            undetected_recovering[:, :, t + 1] = np.maximum(undetected_recovering[:, :, t + 1], 0)

            deceased[:, :, t + 1] = deceased[:, :, t] + self.death_rate[:, None] * (
                    hospitalized_dying[:, :, t] + quarantined_dying[:, :, t] # + undetected_dying[:, :, t]
            ) * self.days_per_timestep

            recovered[:, :, t + 1] = recovered[:, :, t] + (
                    self.hospitalized_recovery_rate * hospitalized_recovering[:, :, t]
                    + self.unhospitalized_recovery_rate * (quarantined_recovering[:, :, t]) # + undetected_recovering[:, :, t])
            ) * self.days_per_timestep

        return DELPHISolution(
            susceptible=susceptible,
            exposed=exposed,
            infectious=infectious,
            hospitalized_dying=hospitalized_dying,
            hospitalized_recovering=hospitalized_recovering,
            quarantined_dying=quarantined_dying,
            quarantined_recovering=quarantined_recovering,
            undetected_dying=undetected_dying,
            undetected_recovering=undetected_recovering,
            recovered=recovered,
            deceased=deceased,
            eligible=eligible,
            vaccinated=vaccinated,
            locations=locations,
            location_indicator=location_indicator,
            vaccine_distribution=vaccine_distribution,
            days_per_timestep=self.days_per_timestep,
        )

    def optimize_relaxation(
            self,
            exploration_tol: float,
            estimated_infectious: np.ndarray,
            locations_per_state_warm_start: Optional[np.ndarray],
            vaccinated_warm_start: Optional[np.ndarray],
            mip_gap: Optional[float],
            feasibility_tol: Optional[float],
            time_limit: Optional[float],
            output_flag: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve a linear relaxation of the vaccine allocation problem, based on estimated infectious populations.

        :param exploration_tol: a positive float that specifies maximum allowed absolute error between the
        estimated and actual infectious population in any region
        :param estimated_infectious: numpy array of size (n_regions, k_classes, t_timesteps + 1) that represents the
        estimated infectious population by region and risk class at each timestep, based on a previous feasible
        solution
        :param vaccinated_warm_start: an optional numpy array of size (n_regions, k_classes, t_timesteps + 1) that
        represents the initial guess for the optimal vaccine allocation policy
        :param mip_gap: an optional float that if set overrides Gurobi's default maximum MIP gap required for
        termination
        :param feasibility_tol: an optional float that if set overrides Gurobi's default maximum feasibility tolerance
        for constraints
        :param time_limit: an optional float that if set specifies the maximum solve time in seconds
        :param output_flag: a boolean that specifies whether to show the solver logs
        :return: a tuple of two numpy arrays of size (n_regions, n_risk_classes, t_timesteps + 1) and (n_regions,) that
        respectively represent the vaccine allocation policy and the regional allocation capacities
        allocations
        """
        # Initialize model
        model = gp.Model()

        # Define decision variables
        susceptible = model.addVars(self.n_regions, self.n_risk_classes, self.n_timesteps + 1, lb=0)
        exposed = model.addVars(self.n_regions, self.n_risk_classes, self.n_timesteps + 1, lb=0)
        infectious = model.addVars(self.n_regions, self.n_risk_classes, self.n_timesteps + 1, lb=0)
        hospitalized_dying = model.addVars(self.n_regions, self.n_risk_classes, self.n_timesteps + 1, lb=0)
        quarantined_dying = model.addVars(self.n_regions, self.n_risk_classes, self.n_timesteps + 1, lb=0)
        undetected_dying = model.addVars(self.n_regions, self.n_risk_classes, self.n_timesteps + 1, lb=0)
        deceased = model.addVars(self.n_regions, self.n_risk_classes, self.n_timesteps + 1, lb=0)

        vaccinated = model.addVars(self.n_regions, self.n_risk_classes, self.n_timesteps + 1, lb=0)
        eligible = model.addVars(self.n_regions, self.n_risk_classes, self.n_timesteps + 1, lb=0)
#         infectious_error = model.addVars(self.n_regions, self.n_timesteps, lb=0) # DROP
#         unallocated_vaccines = model.addVars(self.n_timesteps, lb=0) # DROP
        locations_per_state = model.addVars(self.n_regions, vtype=GRB.INTEGER)
        location_indicator = model.addVars(self.n_cities, vtype=GRB.BINARY)
        # Vaccine distribution variables: C_{it}
        vaccine_distribution = model.addVars(self.n_cities, self.n_timesteps + 1, lb=0)
#        county_city_indicator =  model.addVars(self.n_counties, self.n_cities, lb=0, ub=1)
#        max_center_density = model.addVar(lb=0)
#        min_center_density = model.addVar(lb=0)

        if locations_per_state_warm_start.size > 0:
            for j in self.regions:
                locations_per_state[j].start = locations_per_state_warm_start[j]
                binaries_set = 0
                for i in self.state_to_cities[j]:
                    if binaries_set < locations_per_state_warm_start[j]:
                        location_indicator[i].start = 1
                    else:
                        location_indicator[i].start = 0
                    binaries_set += 1

        # Dummy constraint: consistency between locations_per_state and location_indicator
        model.addConstrs(
            sum(location_indicator[i] for i in self.state_to_cities[j]) == locations_per_state[j] # TODO: check state_to_cities
            for j in self.regions
        )

        # Set initial conditions for DELPHI model
        model.addConstrs(
            susceptible[j, k, 0] == self.initial_susceptible[j, k]
            for j in self.regions for k in self.risk_classes
        )
        model.addConstrs(
            exposed[j, k, 0] == self.initial_exposed[j, k]
            for j in self.regions for k in self.risk_classes
        )
        model.addConstrs(
            infectious[j, k, 0] == self.initial_infectious[j, k]
            for j in self.regions for k in self.risk_classes
        )
        model.addConstrs(
            hospitalized_dying[j, k, 0] == self.initial_hospitalized_dying[j, k]
            for j in self.regions for k in self.risk_classes
        )
        model.addConstrs(
            quarantined_dying[j, k, 0] == self.initial_quarantined_dying[j, k]
            for j in self.regions for k in self.risk_classes
        )
        model.addConstrs(
            undetected_dying[j, k, 0] == self.initial_undetected_dying[j, k]
            for j in self.regions for k in self.risk_classes
        )
        model.addConstrs(
            deceased[j, k, 0] == 0
            for j in self.regions for k in self.risk_classes
        )

        # Set terminal conditions for vaccinations
        model.addConstrs(
            vaccinated[j, k, self.n_timesteps] == 0
            for j in self.regions for k in self.risk_classes
        )

        # Set facility location constraints
        model.addConstr(locations_per_state.sum("*") == self.cities_budget)

        model.addConstrs(
            locations_per_state[j] >= 1 for j in self.regions
        )

        # Fairness (vassilis): balanced location
        model.addConstrs(
            locations_per_state[j] <= self.state_population[j,:].sum() / self.state_population.sum() * self.cities_budget + self.balanced_location for j in self.regions
        )

        model.addConstrs(
            locations_per_state[j] >= self.state_population[j,:].sum() / self.state_population.sum() * self.cities_budget - self.balanced_location for j in self.regions
        )

        # Fairness (michael): center density
#        model.addConstrs(
#            locations_per_state[j] / (self.state_population[j,:].sum() / 1e5) <= max_center_density for j in self.regions
#        )
#
#        model.addConstrs(
#            locations_per_state[j] / (self.state_population[j,:].sum() / 1e5) >= min_center_density for j in self.regions
#        )

        # Vaccine budget
        model.addConstrs(
            vaccine_distribution.sum("*", t) <= self.vaccine_budget[t]
            for t in range(self.n_timesteps)
        )

        # Vaccine distribution: linking constraints (consistency)
        model.addConstrs(
            vaccine_distribution[i, t] <= self.vaccine_budget[t] * location_indicator[i]
            for i in range(self.n_cities) for t in range(self.n_timesteps)
        )

        model.addConstrs(
            sum(vaccinated[j, k, t] for k in self.risk_classes) <= sum(vaccine_distribution[i, t] for i in self.state_to_cities[j])
            for j in self.regions for t in range(self.n_timesteps)
        )

#         model.addConstrs(
#             vaccine_distribution.sum("*", t) == self.vaccine_budget[t]
#             for t in range(self.n_timesteps)
#         )

#         model.addConstrs(
#             sum(vaccinated[j, k, t] for j in self.regions for k in self.risk_classes) == self.vaccine_budget[t]
#             for t in range(self.n_timesteps)
#         )

        # Vaccine distribution: proportionality with state population
        model.addConstrs(
            sum(vaccine_distribution[i, t] for i in self.state_to_cities[j]) <= (self.state_population[j,:].sum() / self.state_population.sum() + self.population_equity_pct) * self.vaccine_budget[t]
            for j in self.regions for t in range(self.n_timesteps)
        )

        # Vaccine distribution: balanced distribution across locations
        model.addConstrs(
            vaccine_distribution[i, t] <= self.vaccine_budget[t]/self.cities_budget * (1 + self.balanced_distr_locations_pct)
            for i in range(self.n_cities) for t in range(self.n_timesteps)
        )

        model.addConstrs(
             vaccine_distribution[i, t] >= (self.vaccine_budget[t]/self.cities_budget * 1 / (1 + self.balanced_distr_locations_pct)) * location_indicator[i]
             for i in range(self.n_cities) for t in range(self.n_timesteps)
        )

        # Sanity check: plan b constraint
        # model.addConstrs(
        #     sum(vaccine_distribution[i, t] for i in self.state_to_cities[j]) == self.vaccine_budget[t]/100*locations_per_state[j]
        #     for j in self.regions for t in range(self.n_timesteps)
        # )

        # Set DELPHI dynamics constraints
        model.addConstrs(
            susceptible[j, k, t + 1] - susceptible[j, k, t] + self.vaccine_effectiveness * vaccinated[j, k, t] >=
            - self.infection_rate[j] * self.policy_response[j, t] / self.state_population[j, :].sum()
            * (susceptible[j, k, t] - self.vaccine_effectiveness * vaccinated[j, k, t])
            * estimated_infectious[j, t] * self.days_per_timestep
            for j in self.regions for k in self.risk_classes for t in self.timesteps
        )

        model.addConstrs(
            exposed[j, k, t + 1] - exposed[j, k, t] >= (
                    self.infection_rate[j] * self.policy_response[j, t] / self.state_population[j, :].sum()
                    * (susceptible[j, k, t] - self.vaccine_effectiveness * vaccinated[j, k, t])
                    * estimated_infectious[j, t]
                    - self.progression_rate * exposed[j, k, t]
            ) * self.days_per_timestep
            for j in self.regions for k in self.risk_classes for t in self.timesteps
        )

        model.addConstrs(
            infectious[j, k, t + 1] - infectious[j, k, t] >= (
                    self.progression_rate * exposed[j, k, t]
                    - self.detection_rate * infectious[j, k, t]
            ) * self.days_per_timestep
            for j in self.regions for k in self.risk_classes for t in self.timesteps
        )

        model.addConstrs(
            hospitalized_dying[j, k, t + 1] - hospitalized_dying[j, k, t] >= (
                    self.ihd_transition_rate[j, k, t] * infectious[j, k, t]
                    - self.death_rate[j] * hospitalized_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self.regions for k in self.risk_classes for t in self.timesteps
        )

        model.addConstrs(
            quarantined_dying[j, k, t + 1] - quarantined_dying[j, k, t] >= (
                    self.iqd_transition_rate[j, k, t] * infectious[j, k, t]
                    - self.death_rate[j] * quarantined_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self.regions for k in self.risk_classes for t in self.timesteps
        )

        model.addConstrs(
            undetected_dying[j, k, t + 1] - undetected_dying[j, k, t] >= (
                    self.iud_transition_rate[j, k, t] * infectious[j, k, t]
                    - self.death_rate[j] * undetected_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self.regions for k in self.risk_classes for t in self.timesteps
        )

        model.addConstrs(
            deceased[j, k, t + 1] - deceased[j, k, t] >= self.death_rate[j] * (
                    hospitalized_dying[j, k, t] + quarantined_dying[j, k, t] # + undetected_dying[j, k, t]
            ) * self.days_per_timestep
            for j in self.regions for k in self.risk_classes for t in self.timesteps
        )

        # Set bounding constraint on absolute error of estimated infectious
        # model.addConstrs(
        #     infectious_error[j, t] <= exploration_tol
        #     for j in self.regions for t in self.timesteps
        # )
        # model.addConstrs(
        #     infectious_error[j, t] >= estimated_infectious[j, t] - infectious.sum(j, "*", t)
        #     for j in self.regions for t in self.timesteps
        # )
        # model.addConstrs(
        #     infectious_error[j, t] >= infectious.sum(j, "*", t) - estimated_infectious[j, t]
        #     for j in self.regions for t in self.timesteps
        # )

        # Set eligibility constraints
        model.addConstrs(
            eligible[j, k, t + 1] <= eligible[j, k, t] - (1 - self.vaccine_effectiveness)
            * vaccinated[j, k, t] - (susceptible[j, k, t] - susceptible[j, k, t + 1])
            for j in self.regions for k in self.included_risk_classes for t in self.timesteps
        )

        model.addConstrs(
            vaccinated[j, k, t] <= eligible[j, k, t]
            for j in self.regions for k in self.included_risk_classes for t in self.timesteps
        )

        # Set risk-class vaccination constraints
        model.addConstrs( # TODO: consider removing this or changing excluded_risk_classes
            vaccinated[j, k, t] == 0
            for j in self.regions for k in self.excluded_risk_classes for t in self.timesteps
        )

        # Vaccine smoothing constraints
        model.addConstrs(
            vaccine_distribution[i, t + 1] >= (1-self.max_distr_pct_change)*vaccine_distribution[i, t]
            for i in range(self.n_cities) for t in self.timesteps[:-1]
        )

        model.addConstrs(
            vaccine_distribution[i, t + 1] <= (1+self.max_distr_pct_change)*vaccine_distribution[i, t]
            for i in range(self.n_cities) for t in self.timesteps[:-1]
        )

#         model.addConstrs(
#             unallocated_vaccines[t] >= self.vaccine_budget[t] - vaccinated.sum("*", "*", t)
#             for t in self.timesteps
#         )
        
        # Set conditions for counties being assigned to a city
        
        # we can only assign counties to cities that are being chosen
#        model.addConstrs(
#            county_city_indicator[l,i] <= location_indicator[i]
#            for l in range(self.n_counties) for i in range(self.n_cities)
#        )
        
        # all counties needs to be assigned to one county
#        model.addConstrs(
#            gp.quicksum(county_city_indicator[l,i] for i in range(self.n_cities)) == 1
#            for l in range(self.n_counties)
#        )

        # Objective
        model.setObjective(
            deceased.sum("*", "*", self.n_timesteps)
            + hospitalized_dying.sum("*", "*", self.n_timesteps)
            + quarantined_dying.sum("*", "*", self.n_timesteps)
            - self.vaccination_enforcement_weight * vaccinated.sum("*", "*","*")
            # + undetected_dying.sum("*", "*", self.n_timesteps)
#             + unallocated_vaccines.sum()
#            + self.political_factor * (max_center_density - min_center_density)
#            + self.distance_penalty * gp.quicksum(county_city_indicator[l,i] * self.population[l,:].sum() * self.county_city_to_distance[(l,i)] for l, i in self.county_city_to_distance.keys())
            ,
            GRB.MINIMIZE
        )

        # Set solver params
        if mip_gap:
            model.params.MIPGap = mip_gap

        if feasibility_tol:
            model.params.FeasibilityTol = feasibility_tol

        if time_limit:
            model.params.TimeLimit = time_limit

        model.params.OutputFlag = True
        model.params.TimeLimit = 600
        model.params.method = 2 ######## Barrier for root
        # model.params.nodeMethod = 2 ######## Barrier for subsequent nodes

        # Solve model
        model.optimize()
        if model.status == 2:
            print(f"\nPARAMS:")
    #        print(f"Ratio: {max_center_density.x/min_center_density.x}")
            print(f"Political Factor: {self.political_factor}")
            print(f"Balanced location: {self.balanced_location}")
            print(f"Max distr pct change: {self.max_distr_pct_change}")
            print(f"Population equity pct: {self.population_equity_pct}")
            print(f"Vaccination enforcement weight: {self.vaccination_enforcement_weight}")
            print(f"Balanced location distribution pct: {self.balanced_distr_locations_pct}")

            # Return vaccine allocation
            vaccinated = model.getAttr("x", vaccinated)
            vaccinated = np.array([
                [[vaccinated[j, k, t] for t in range(self.n_timesteps + 1)] for k in self.risk_classes]
                for j in self.regions
            ])
            locations = model.getAttr("x", locations_per_state)
            locations = np.array([locations[j] for j in self.regions])

            location_indicator = model.getAttr("x", location_indicator)
            location_indicator = np.array([location_indicator[i] for i in range(self.n_cities)])

            vaccine_distribution = model.getAttr("x", vaccine_distribution)
            vaccine_distribution = np.array([[vaccine_distribution[i, t] for t in range(self.n_timesteps + 1)] for i in range(self.n_cities)])



            minimum_density_state = np.argmin(np.divide(locations,self.state_population.sum(axis=1)))
            maximum_density_state = np.argmax(np.divide(locations,self.state_population.sum(axis=1)))
            minimum_density = np.min(np.divide(locations,self.state_population.sum(axis=1))) * 1e6
            maximum_density = np.max(np.divide(locations,self.state_population.sum(axis=1))) * 1e6
            print(f"Minimum density: {minimum_density_state}")
            print(f"Maximum density: {maximum_density_state}")
            print(f"Maximum density: {maximum_density}")
            print(f"Minimum density: {minimum_density}")
        else:
            vaccinated = None
            locations = None
            location_indicator = None
            vaccine_distribution = None
        return vaccinated, locations, location_indicator, vaccine_distribution

    def smooth_vaccine_allocation(
            self,
            solution: DELPHISolution,
            smoothing_window: int
    ) -> DELPHISolution:
        """
        Apply a rolling-window smoothing heuristic to the vaccine allocation policy as a post-processing step.

        :param solution: a DELPHISolution object
        :param smoothing_window: an integer that specifies the symmetric smoothing window size (default)
        :return: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1) representing the smoothed vaccine
        allocation policy
        """
        vaccinated = np.zeros(solution.vaccinated.shape)
        for t in self.timesteps:
            start = max(t - smoothing_window, 0)
            end = min(t + smoothing_window, self.n_timesteps) + 1
            vaccinated[:, :, t] = solution.vaccinated[:, :, start:end].mean(axis=2)
            if vaccinated[:, :, t].sum() > 0:
                vaccinated[:, :, t] = vaccinated[:, :, t] * self.vaccine_budget[t] / vaccinated[:, :, t].sum()
            else:
                vaccinated[:, :, t] = 0
        return self.simulate(vaccinated=vaccinated, locations=solution.locations)

    def prioritize_vaccine_allocation(self, solution: DELPHISolution) -> DELPHISolution:
        """
        Adjust vaccine allocation to ensure risk class prioritization is preserved.

        :param solution: a DELPHISolution object
        :return: a DELPHISolution object
        """

        # Rank risk classes by mortality rate
        priority_ranking = np.argsort(-self.mortality_rate.mean(axis=(0, 2)))

        for t in self.timesteps:

            # Initialize variables for timestep
            eligible = solution.eligible
            vaccinated = solution.vaccinated
            update = False

            # Perform vaccine transfers to prioritize by risk class
            for i, k in enumerate(priority_ranking):
                for l in priority_ranking[i + 1:]:
                    if k in self.included_risk_classes and l in self.included_risk_classes:
                        transfer = np.minimum(vaccinated[:, l, t], eligible[:, k, t] - vaccinated[:, k, t])
                        if transfer.max() > 0:
                            vaccinated[:, k, t] = vaccinated[:, k, t] + transfer
                            vaccinated[:, l, t] = vaccinated[:, l, t] - transfer
                            update = True

            # If a transfer was
            if update:
                solution = self.simulate(vaccinated=vaccinated, locations=solution.locations)

        return solution

    def round_vaccine_allocation(self, solution: DELPHISolution, rounding_tol: float):
        """
        Apply a rounding cut-off to the vaccine allocation policy as a proxy for transforming the interior point
        solution into a basic feasible solution.

        :param solution: a DELPHISolution object
        :param rounding_tol: a float the specifies the maximum allocation amount that will be rounded to 0 in
        post-processing (default 1e-3)
        :return: a DELPHISolution object
        """
        vaccinated = np.where(solution.vaccinated > rounding_tol, solution.vaccinated, 0)
        for t in self.timesteps:
            if vaccinated[:, :, t].sum() > 0:
                vaccinated[:, :, t] = vaccinated[:, :, t] * self.vaccine_budget[t] / vaccinated[:, :, t].sum()
            else:
                vaccinated[:, :, t] = 0
        return self.simulate(vaccinated=vaccinated, locations=solution.locations)

    def post_process_solution(
            self,
            solution: DELPHISolution,
            prioritize_allocation: bool,
            smooth_allocation: bool,
            round_allocation: bool,
            smoothing_window: int,
            rounding_tol: float
    ):
        """

        :param solution: a DELPHISolution object
        :param smoothing_window: an integer that specifies the symmetric smoothing window size (default)
        :param rounding_tol: a float the specifies the maximum allocation amount that will be rounded to 0 in
        post-processing (default 1e-3)
        :return: a numpy array of size (n_regions, n_risk_classes, n_timesteps + 1) representing the smoothed vaccine
        allocation policy
        """
        if smooth_allocation:
            solution = self.smooth_vaccine_allocation(solution=solution, smoothing_window=smoothing_window)
        if prioritize_allocation:
            solution = self.prioritize_vaccine_allocation(solution=solution)
        if round_allocation:
            solution = self.round_vaccine_allocation(solution=solution, rounding_tol=rounding_tol)
        return solution

    def optimize(
            self,
            exploration_tol: float,
            termination_tol: float = 1e-2,
            mip_gap: Optional[float] = 1e-2,
            feasibility_tol: Optional[float] = None,
            time_limit: Optional[float] = None,
            output_flag: bool = False,
            n_restarts: int = 1,
            max_iterations: int = 10,
            n_early_stopping_iterations: int = 2,
            smooth_allocation: bool = False,
            prioritize_allocation: bool = False,
            round_allocation: bool = False,
            smoothing_window: int = 1,
            rounding_tol: float = 1e-2,
            log: bool = False,
            seed: int = 0,
            fixed_cities: bool = False
    ) -> DELPHISolution:
        """
        Solve the prescriptive DELPHI model for vaccine allocation using a coordinate descent heuristic.

        :param exploration_tol: a float  that specifies maximum allowed absolute error between the
        estimated and actual infectious population in any region
        :param termination_tol: a positive float that specifies maximum allowed absolute error between the
        estimated and actual infectious population in any region (default 1e-3)
        :param mip_gap: an optional float that if set overrides Gurobi's default maximum MIP gap required for
        termination (default None)
        :param feasibility_tol: an optional float that if set overrides Gurobi's default maximum feasibility tolerance
        for constraints (default None)
        :param time_limit: an optional float that if set specifies the maximum solve time in seconds (default None)
        :param round_allocation: something
        :param output_flag: a boolean that specifies whether to show the solver logs (default False)
        :param n_restarts: an integer that specifies the number of restarts, with a smart start provided if set to 1
        else randomized starts provided (default 1)
        :param max_iterations: an integer that specifies the maximum number of descent iterations per trial (default 10)
        :param n_early_stopping_iterations: an integer that specifies the number of descent iterations after which a
        trial is terminated if there is no improvement (default 2)
        :param smooth_allocation: a boolean that if true applies an additional smoothing heuristic to the solution in
        post-processing(default True)
        :param prioritize_allocation: a boolean
        :param smoothing_window: an integer that specifies the symmetric smoothing window size (default)
        :param rounding_tol: a float the specifies the maximum allocation amount that will be rounded to 0 in
        post-processing (default 1e-3)
        :param log: a boolean that specifies whether to log progress
        :param seed: an integer that is used to set the numpy seed
        :param fixed_cities:
        :return: a DELPHISolution object
        """

        # Initialize algorithm
        np.random.seed(seed)
        best_solution = None
        best_obj_val = np.inf

        for restart in range(n_restarts):

            # Initialize restart
            incumbent_solution = self.simulate(
                randomize_allocation=False,
                prioritize_allocation=False,
                initial_solution_allocation=True
            )
            incumbent_obj_val = incumbent_solution.get_total_deaths()
            trajectory = [incumbent_obj_val]
            best_solution_for_restart = None
            best_obj_val_for_restart = np.inf
            n_iters_since_improvement = 0

            if log:
                print(f"Restart: {restart + 1}/{n_restarts}")
                print(f"Iteration: 0/{max_iterations} \t Objective value: {'{0:.2f}'.format(incumbent_obj_val)}")

            for i in range(max_iterations):

                # Re-optimize vaccine allocation by solution linearized relaxation
                try:
                    vaccinated, locations, location_indicator, vaccine_distribution = self.optimize_relaxation(
                        exploration_tol=exploration_tol,
                        estimated_infectious=incumbent_solution.infectious.sum(axis=1),
                        vaccinated_warm_start=incumbent_solution.vaccinated,
                        locations_per_state_warm_start=incumbent_solution.locations,
                        mip_gap=mip_gap,
                        feasibility_tol=feasibility_tol,
                        time_limit=time_limit,
                        output_flag=output_flag,
                    )

                except GurobiError as error:
                    print(error)
                    if log:
                        print("Infeasible relaxation - terminating search")
                    break
                if vaccinated is None:
                    break

                # Update incumbent and previous solution
                previous_solution, incumbent_solution = incumbent_solution, self.simulate(vaccinated=vaccinated,
                                                                                          locations=locations,
                                                                                          location_indicator=location_indicator,
                                                                                          vaccine_distribution=vaccine_distribution)
                previous_obj_val, incumbent_obj_val = incumbent_obj_val, incumbent_solution.get_total_deaths()
                trajectory.append(incumbent_obj_val)
                if log:
                    print(
                        f"Iteration: {i + 1}/{max_iterations} \t Objective value: {'{0:.2f}'.format(incumbent_obj_val)}")

                # Update best solution if incumbent solution is an improvement
                if incumbent_obj_val < best_obj_val:
                    best_obj_val = incumbent_obj_val
                    best_solution = incumbent_solution

                # Update incumbent solution for restart if incumbent solution is an improvement
                if incumbent_obj_val < best_obj_val_for_restart:
                    best_obj_val_for_restart = incumbent_obj_val
                    best_solution_for_restart = incumbent_solution
                    n_iters_since_improvement = 0
                else:
                    n_iters_since_improvement += 1

                # Terminate coordinate descent for restart if solution convergences
                if n_iters_since_improvement >= n_early_stopping_iterations:
                    if log:
                        print(
                            f"No improvement found in {n_early_stopping_iterations} iterations"
                            f" - terminating search for trial"
                        )
                    break

                # Terminate coordinate descent for restart if solution convergences
                objective_change = abs(previous_obj_val - incumbent_obj_val)
                estimated_infectious_change = np.abs(
                    previous_solution.infectious.sum(axis=1) - incumbent_solution.infectious.sum(axis=1)
                ).mean()
                if max(objective_change, estimated_infectious_change) < termination_tol:
                    trajectory.append(incumbent_obj_val)
                    if log:
                        print("Solution has converged - terminating search for trial")
                    break

            # Store trajectory and best solution for completed restart
            self.trajectories.append(trajectory)
            self.solutions.append(best_solution_for_restart)

        if best_solution is None:
            best_solution = incumbent_solution

        # Apply post-processing steps to solution
        if log:
            print("Post-processing solution")
        best_solution = self.post_process_solution(
            solution=best_solution,
            smooth_allocation=smooth_allocation,
            prioritize_allocation=prioritize_allocation,
            round_allocation=round_allocation,
            smoothing_window=smoothing_window,
            rounding_tol=rounding_tol
        )
        if log:
            print(f"Objective value after post-processing: {'{0:.2f}'.format(best_solution.get_total_deaths())}")
        return best_solution
