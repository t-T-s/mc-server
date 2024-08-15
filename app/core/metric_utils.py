import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from app.utils.generic import image_buffer_matplotlib, image_buffer_plotly


# Consistency calculation with shapash
def calc_all_consistency_scores(cns, contributions):
    _, mean_distances = cns.calculate_all_distances(cns.methods, cns.weights)
    ind_avg = []
    for col in mean_distances.columns:
        ind_avg.append(sum(mean_distances[col]) / 2)
    avg = np.average(ind_avg)

    contrib_keys = list(contributions.keys())
    combinations_ls = list(combinations(contrib_keys, 2))
    consistency_scores = {}
    for comb in combinations_ls:
        comb_ls = list(comb)
        comb_ls.sort()
        consistency_scores["_".join(comb_ls)] = mean_distances.loc[comb]
    return avg, consistency_scores


# Convert contributions list to pandas DataFrame
def convert_contrib_dict_to_dataframe(contributions: dict):
    for key, value in contributions.items():
        contributions[key] = pd.DataFrame(value)
    return contributions


def convert_array_like_to_dataframe(contributions: list):
    pd_contributions = pd.DataFrame(contributions)
    return pd_contributions


# create dummy model for smart explainer
def create_dummy_model():
    x_train = [[0, 5, 6, 4], [1, 1, 9, 4]]
    y_train = [0, 1]
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    rf.fit(x_train, y_train)
    return rf


def train_surrogate_rf_model(x, y, **kwargs):
    rf = RandomForestClassifier(**kwargs)
    rf.fit(x, y)
    return rf


def get_one_minus_distance_reached(xpl):
    return np.mean(1 - xpl.features_compacity['distance_reached'])


def plot_consistency(cns):
    """
    Plot consistency values
    :param cns: Consistency object
    :return: matplotlib figure
    """
    _, mean_distances = cns.calculate_all_distances(cns.methods, cns.weights)
    fig = cns.plot_comparison(mean_distances)
    return fig


def plot_compacity(xpl, selection, approx=0.9, nb_features=5):
    fig = xpl.plot.compacity_plot(selection=selection, approx=approx, nb_features=nb_features)
    return fig


def plot_stability(xpl, selection, max_points=500, max_features=10):
    fig = xpl.plot.stability_plot(selection=selection, max_points=max_points, max_features=max_features)
    return fig


def buffer_plot(fig):
    img_buffer = None
    if isinstance(fig, plt.Figure):
        img_buffer = image_buffer_matplotlib(fig, img_type='png')
        plt.close(fig)
    elif isinstance(fig, go.Figure):
        img_buffer = image_buffer_plotly(fig, img_type='png')
    return img_buffer
