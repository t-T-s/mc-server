import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier


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
def convert_contrib_dict_to_dataframe(contributions):
    for key, value in contributions.items():
        contributions[key] = pd.DataFrame(value)
    return contributions


def convert_contrib_to_dataframe(contributions: list):
    pd_contributions = pd.DataFrame(contributions)
    return pd_contributions


# create dummy model for smart explainer
def create_dummy_model():
    X_train = [[0, 5, 6], [1, 1, 9]]
    y_train = [0, 1]
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    return rf


def get_one_minus_distance_reached(xpl):
    return np.mean(1 - xpl.features_compacity['distance_reached'])
