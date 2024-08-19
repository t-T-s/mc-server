from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from shapash.explainer.consistency import Consistency
from shapash import SmartExplainer
from app.core import metric_utils
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


# Here the metrics will be implemented with the external libraries.


def scikit_accuracy(ground_truth, predictions) -> float:
    return accuracy_score(ground_truth, predictions)


def shapash_consistency(contributions_dict):
    # The format must be a dictionary where keys are the methods
    # names and values are pandas DataFrames: be careful to have dataframes
    # with same shape, index and column names
    cns = Consistency()
    pd_contributions = metric_utils.convert_contrib_dict_to_dataframe(contributions_dict)
    cns.compile(contributions=pd_contributions)
    avg_consistency, pairwise_consistencies = metric_utils.calc_all_consistency_scores(cns, pd_contributions)
    return avg_consistency, pairwise_consistencies


def shapash_compacity_from_contributions(contributions, selection=None, distance=0.9, nb_features=5):
    """
    model: dummy model
    contributions: contributions as a list of lists
    selection: a sample of the dataset on which to evaluate the metric expressed
    as a list of indices (by default take the whole dataset if not too big)
    list of indices of datapoints (The length of the output lists will be equal
    to the length of the selection list)
    distance (float): how close we want to be the reference model with all features (default 90%)
                     – Left graph
    nb_features (int): how many features are selected to evaluate the approximation (default 5)
                    – Right graph

    return: dict of features compacity
    """
    model = metric_utils.create_dummy_model()
    if selection is None:
        selection = list(range(len(contributions)))
    pd_contributions = metric_utils.convert_array_like_to_dataframe(contributions)
    # Not the best method to do this, but it works. Later we can implement in a cleaner way.
    xpl = SmartExplainer(model=model)
    xpl._get_contributions_from_backend_or_user(x=None, contributions=pd_contributions)
    xpl.compute_features_compacity(selection=selection
                                   , distance=distance
                                   , nb_features=nb_features)
    xpl.features_compacity['distance_reached'] = xpl.features_compacity['distance_reached'].tolist()
    return xpl.features_compacity


def shapash_consistency_plot(contributions_dict):
    # The format must be a dictionary where keys are the methods
    # names and values are pandas DataFrames: be careful to have dataframes
    # with same shape, index and column names
    cns = Consistency()
    pd_contributions = metric_utils.convert_contrib_dict_to_dataframe(contributions_dict)
    cns.compile(contributions=pd_contributions)
    # avg_consistency, pairwise_consistencies = metric_utils.calc_all_consistency_scores(cns, pd_contributions)
    comparison_plot = metric_utils.plot_consistency(cns)
    return comparison_plot


def shapash_compacity_plot(contributions, selection=None, distance=0.9, nb_features=5):
    """
    model: dummy model
    contributions: contributions as a list of lists
    selection: a sample of the dataset on which to evaluate the metric expressed
    as a list of indices (by default take the whole dataset if not too big)
    list of indices of datapoints (The length of the output lists will be equal
    to the length of the selection list)
    distance (float): how close we want to be the reference model with all features (default 90%)
                     – Left graph
    nb_features (int): how many features are selected to evaluate the approximation (default 5)
                    – Right graph

    return: dict of features compacity
    """
    model = metric_utils.create_dummy_model()
    if selection is None:
        selection = list(range(len(contributions)))
    pd_contributions = metric_utils.convert_array_like_to_dataframe(contributions)
    # Not the best method to do this, but it works. Later we can implement in a cleaner way.
    xpl = SmartExplainer(model=model)
    xpl._get_contributions_from_backend_or_user(x=None, contributions=pd_contributions)
    compacity_plot = metric_utils.plot_compacity(xpl, selection=selection
                                                 , approx=distance
                                                 , nb_features=nb_features)
    return compacity_plot


def shapash_stability_plot(x_test,
                           contributions,
                           y_target,
                           selection=None,
                           max_points=500,
                           max_features=10,
                           model=None,
                           x_train=None,
                           y_train=None):
    """
    model: dummy model
    contributions: contributions as a list of lists
    selection: a sample of the dataset on which to evaluate the metric expressed
    as a list of indices (by default take the whole dataset if not too big)
    list of indices of datapoints (The length of the output lists will be equal
    to the length of the selection list)
    distance (float): how close we want to be the reference model with all features (default 90%)
                     – Left graph
    nb_features (int): how many features are selected to evaluate the approximation (default 5)
                    – Right graph

    return: dict of features compacity
    """
    if model is None:
        if (x_train is not None) and (x_test is not None):
            model = metric_utils.train_surrogate_rf_model(x=x_train, y=y_train, n_estimators=100, max_depth=10,
                                                      random_state=0)
        else:
            raise AttributeError("x_train and x_test must be available to train a surrogate model if a model is not "
                                 "provided.")
    # Not the best method to do this, but it works. Later we can implement in a cleaner way.
    xpl = SmartExplainer(model=model)
    # to plot the stability plot, the model is not necessary if you have the y_targets as well.
    xpl.compile(x=metric_utils.convert_array_like_to_dataframe(x_test)
                , contributions=metric_utils.convert_array_like_to_dataframe(contributions)
                , y_target=metric_utils.convert_array_like_to_dataframe(y_target))
    stability_plot = metric_utils.plot_stability(xpl
                                                 , selection
                                                 , max_points=max_points
                                                 , max_features=max_features
                                                 )
    return stability_plot


# def shapash_feature_stability(x_encoded, contributions, y_target, selection=None, max_points=500):
#     """
#     model: dummy model
#     contributions: contributions as a list of lists
#     selection: a sample of the dataset on which to evaluate the metric expressed
#     as a list of indices (by default take the whole dataset if not too big)
#     list of indices of datapoints (The length of the output lists will be equal
#     to the length of the selection list)
#     distance (float): how close we want to be the reference model with all features (default 90%)
#                      – Left graph
#     nb_features (int): how many features are selected to evaluate the approximation (default 5)
#                     – Right graph
#
#     return: dict of features compacity
#     """
#     model = metric_utils.create_dummy_model()
#     # Not the best method to do this, but it works. Later we can implement in a cleaner way.
#     xpl = SmartExplainer(model=model)
#     ## to plot the stability plot, the model is not necessary if you have the y_targets as well.
#     xpl.compile(x=metric_utils.convert_array_like_to_dataframe(x_encoded)
#                 , contributions=metric_utils.convert_array_like_to_dataframe(contributions)
#                 , y_target=metric_utils.convert_array_like_to_dataframe(y_target))
#     print("x_init: ", xpl.x_init)
#     print("xpl_feature_stability_after_compile: ", xpl.features_stability)
#     if selection is None:
#         if xpl.x_init.shape[0] <= max_points:
#             list_ind = xpl.x_init.index.tolist()
#             feature_stability = xpl.compute_features_stability(list_ind)
#             print("list_ind: ", list_ind)
#             print("feature_stability_1: ", feature_stability)
#         else:
#             list_ind = random.sample(xpl.x_init.index.tolist(), max_points)
#             feature_stability = xpl.compute_features_stability(list_ind)
#             print("feature_stability_2:", feature_stability)
#     else:
#         feature_stability = xpl.compute_features_stability(selection)
#         print("feature_stability_3:", feature_stability)
#     return feature_stability


def evasion_impact(ground_truth, predictions) -> float:
    return 1 - accuracy_score(ground_truth, predictions)


def tsne_user_diversity(predictions, client_ids, perplexity):
    predictions = np.array(predictions)
    client_ids = np.array(client_ids)
    tsne = TSNE(n_components=2, verbose=0, random_state=123, perplexity=perplexity)
    z = tsne.fit_transform(predictions)
    df = pd.DataFrame()
    df["y"] = client_ids
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    df2 = df.to_json(orient='columns')
    return df2

    # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
    #                 palette=sns.color_palette("hls", 10),
    #                 data=df)
    # plt.legend(fontsize=15, loc='lower left')


def tsne_user_diversity_plot(predictions, client_ids, perplexity):
    predictions = np.array(predictions)
    client_ids = np.array(client_ids)
    tsne = TSNE(n_components=2, verbose=0, random_state=123, perplexity=perplexity)
    z = tsne.fit_transform(predictions)
    df = pd.DataFrame()
    df["y"] = client_ids
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df)
    plt.legend(fontsize=15, loc='lower left')
    return plt.gcf()
