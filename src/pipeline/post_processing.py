# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:16:06 2021

@author: Michael
"""

states = sorted(county_pop_df.state.unique())
with open("../data/outputs/optimized-solution--20210209-200037-0-0.pickle","rb") as f:
    solution = pickle.load(f)
vac_dist_states = np.zeros(solution.vaccinated.shape[0])
for j in range(vac_dist_states.shape[0]):
    vac_dist_states[j] = sum(solution.vaccine_distribution[i,:].sum() for i in allocation_params["state_to_cities"][j])
total_deaths = (solution.deceased + solution.hospitalized_dying + solution.quarantined_dying)[:, :, -1].sum(axis=1)
pd.DataFrame(data={"state": states, "num_locations": solution.locations,"vac_distributions": vac_dist_states, "total_deaths": total_deaths}).to_csv("optimized_all_ok.csv")



with open("../data/outputs/baseline-solution-cities-20210207-153854-0-0.pickle","rb") as f:
    solution = pickle.load(f)
vac_dist_states = solution.locations * 0
total_deaths = (solution.deceased + solution.hospitalized_dying + solution.quarantined_dying)[:, :, -1].sum(axis=1)
pd.DataFrame(data={"state": states, "num_locations": solution.locations,"vac_distributions": vac_dist_states, "total_deaths": total_deaths}).to_csv("no_vaccine.csv")