from app.core.metrics import scikit_accuracy, shapash_consistency, shapash_compacity_from_contributions, evasion_impact
# Here the metrics will be clearly demarcated for different use case.
# e.g.: with plots as outputs, with numbers as outputs, etc.


def clf_accuracy_score(ground_truth, predictions) -> float:
    accuracy = scikit_accuracy(ground_truth, predictions)
    return accuracy


def consistency_scores(contributions_dict):
    # TODO: Implement to get the generated plots (either in the client side
    #  or create plots from the server side)
    return shapash_consistency(contributions_dict)


def compacity_scores(contributions, selection, distance, nb_features):
    # TODO: Implement to get the generated plots (either in the client side
    #  or create plots from the server side)
    return shapash_compacity_from_contributions(contributions=contributions
                                                , selection=selection
                                                , distance=distance
                                                , nb_features=nb_features)


def evasion_impact_score(ground_truth, predictions) -> float:
    return evasion_impact(ground_truth, predictions)
