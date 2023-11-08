from core import scikit_accuracy, shapash_consistency, shapash_compacity, impact
# Here the metrics will be clearly demarcated for different use case.
# e.g.: with plots as outputs, with numbers as outputs, etc.


def clf_accuracy_score(ground_truth, predictions) -> float:
    accuracy = scikit_accuracy(ground_truth, predictions)
    return accuracy


def consistency_scores(contributions):
    # TODO: Implement to get the generated plots (either in the client side
    #  or create plots from the server side)
    return shapash_consistency(contributions)


def compacity_scores(contributions):
    # TODO: Implement to get the generated plots (either in the client side
    #  or create plots from the server side)
    return shapash_compacity(contributions)


def impact_score(contributions):
    pass
