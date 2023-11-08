from sklearn.metrics import accuracy_score
import mc_utils
from shapash.explainer.consistency import Consistency
from shapash import SmartExplainer


# Here the metrics will be implemented with the external libraries.


def scikit_accuracy(ground_truth, predictions) -> float:
    return accuracy_score(ground_truth, predictions)


def shapash_consistency(contributions_dict):
    # TODO: Implement to get the generated plots
    # The format must be a dictionary where keys are the methods
    # names and values are pandas DataFrames: be careful to have dataframes
    # with same shape, index and column names
    cns = Consistency()
    pd_contributions = mc_utils.convert_contrib_dict_to_dataframe(contributions_dict)
    cns.compile(contributions=pd_contributions)
    return mc_utils.calc_all_consistency_scores(cns, pd_contributions)


def shapash_compacity_from_contributions(contributions, selection=None, distance=0.9, nb_features=5):
    # TODO: Implement to get the generated plots

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
    model = mc_utils.create_dummy_model()
    if selection is None:
        selection = list(range(len(contributions)))
    pd_contribs = mc_utils.convert_contrib_to_dataframe(contributions)
    # Not the best method to do this, but it works. Later we can implement in a cleaner way.
    xpl = SmartExplainer(model=model)
    xpl._get_contributions_from_backend_or_user(x=None, contributions=pd_contribs)
    xpl.compute_features_compacity(selection=selection
                                   , distance=distance
                                   , nb_features=nb_features)
    xpl.features_compacity['distance_reached'] = xpl.features_compacity['distance_reached'].tolist()
    return xpl.features_compacity


def impact(contributions):
    pass
