from sklearn.metrics import accuracy_score
from shapash.explainer.consistency import Consistency
from shapash import SmartExplainer
from app.core import metric_utils


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


def shapash_consistency_plot(contributions_dict):
    # The format must be a dictionary where keys are the methods
    # names and values are pandas DataFrames: be careful to have dataframes
    # with same shape, index and column names
    cns = Consistency()
    pd_contributions = metric_utils.convert_contrib_dict_to_dataframe(contributions_dict)
    cns.compile(contributions=pd_contributions)
    avg_consistency, pairwise_consistencies = metric_utils.calc_all_consistency_scores(cns, pd_contributions)
    consistency_plot = metric_utils.plot_consistency(cns)
    consistency_plot_buffer = metric_utils.buffer_plot(consistency_plot)
    return avg_consistency, pairwise_consistencies, consistency_plot_buffer


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
    pd_contributions = metric_utils.convert_contrib_to_dataframe(contributions)
    # Not the best method to do this, but it works. Later we can implement in a cleaner way.
    xpl = SmartExplainer(model=model)
    xpl._get_contributions_from_backend_or_user(x=None, contributions=pd_contributions)
    xpl.compute_features_compacity(selection=selection
                                   , distance=distance
                                   , nb_features=nb_features)
    xpl.features_compacity['distance_reached'] = xpl.features_compacity['distance_reached'].tolist()
    return xpl.features_compacity


def shapash_compacity_plot_from_contributions(contributions, selection=None, distance=0.9, nb_features=5):
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
    pd_contributions = metric_utils.convert_contrib_to_dataframe(contributions)
    # Not the best method to do this, but it works. Later we can implement in a cleaner way.
    xpl = SmartExplainer(model=model)
    xpl._get_contributions_from_backend_or_user(x=None, contributions=pd_contributions)
    fig = metric_utils.plot_compacity(xpl, selection=selection
                                      , approx=distance
                                      , nb_features=nb_features)
    return fig


def evasion_impact(ground_truth, predictions) -> float:
    return 1 - accuracy_score(ground_truth, predictions)
