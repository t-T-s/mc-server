import numpy as np
import pandas as pd
from itertools import combinations


# Consistency calculation with shapash
def calc_all_consistency_scores(cns, contributions):
    _, dist_matrix = cns.calculate_all_distances(cns.methods, cns.weights)
    ind_avg = []
    for col in dist_matrix.columns:
        ind_avg.append(sum(dist_matrix[col]) / 2)
    avg = np.average(ind_avg)

    contrib_keys = list(contributions.keys())
    combinations_ls = list(combinations(contrib_keys, 2))
    consistency_scores = {}
    for comb in combinations_ls:
        comb_ls = list(comb)
        comb_ls.sort()
        consistency_scores["_".join(comb_ls)] = dist_matrix.loc[comb]
    return avg, consistency_scores


# Convert contributions list to pandas DataFrame
def convert_contrib_list_to_dataframe(contributions):
    for key, value in contributions.items():
        contributions[key] = pd.DataFrame(value)
    return contributions
