from sklearn.metrics import accuracy_score
from mc_utils import calc_all_consistency_scores, convert_contrib_list_to_dataframe
from shapash.explainer.consistency import Consistency


def scikit_accuracy(ground_truth, predictions) -> float:
    return accuracy_score(ground_truth, predictions)


def shapash_consistency(contributions):
    # TODO: Implement to get the generated plots
    # The format must be a dictionary where keys are methods
    # names and values are pandas DataFrames: be careful to have dataframes
    # with same shape, index and column names
    cns = Consistency()
    pd_contribs = convert_contrib_list_to_dataframe(contributions)
    cns.compile(contributions=pd_contribs)
    return calc_all_consistency_scores(cns, pd_contribs)


def shapash_compacity(contributions):
    # TODO: Implement to get the generated plots
    pass


def impact(contributions):
    pass
