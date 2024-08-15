from app.core.metrics import scikit_accuracy, shapash_consistency, shapash_compacity_from_contributions, \
    evasion_impact, shapash_consistency_plot, shapash_compacity_plot, tsne_user_diversity, tsne_user_diversity_plot, \
    shapash_stability_plot
from app.core import metric_utils
import joblib
from sklearn.base import BaseEstimator


# Here the metrics will be clearly demarcated for different use case.
# e.g.: with plots as outputs, with numbers as outputs, etc.


def clf_accuracy_score(ground_truth, predictions) -> float:
    accuracy = scikit_accuracy(ground_truth, predictions)
    return accuracy


def consistency_scores(contributions_dict):
    return shapash_consistency(contributions_dict)


def compacity_scores(contributions, selection, distance, nb_features):
    return shapash_compacity_from_contributions(contributions=contributions
                                                , selection=selection
                                                , distance=distance
                                                , nb_features=nb_features)


def consistency_plot(contributions_dict):
    comparison_plot = shapash_consistency_plot(contributions_dict)
    image_buffer = metric_utils.buffer_plot(comparison_plot)
    return image_buffer


def compacity_plot(contributions, selection, distance, nb_features):
    compacity_graph = shapash_compacity_plot(contributions=contributions
                                             , selection=selection
                                             , distance=distance
                                             , nb_features=nb_features)

    image_buffer = metric_utils.buffer_plot(compacity_graph)
    return image_buffer


def stability_plot(x_test, contributions, y_target, selection, max_points, max_features, x_train=None, y_train=None,
                   model_stream=None):
    model: BaseEstimator = joblib.load(model_stream)
    stability_graph = shapash_stability_plot(
        x_train=x_train
        , y_train=y_train
        , model=model
        , x_test=x_test
        , contributions=contributions
        , y_target=y_target
        , selection=selection
        , max_points=max_points
        , max_features=max_features)
    image_buffer = metric_utils.buffer_plot(stability_graph)
    return image_buffer


def evasion_impact_score(ground_truth, predictions) -> float:
    return evasion_impact(ground_truth, predictions)


def user_diversity_score(x_results, y_results, perplexity):
    return tsne_user_diversity(x_results, y_results, perplexity)


def user_diversity_plot(x_results, y_results, perplexity):
    plot = tsne_user_diversity_plot(x_results, y_results, perplexity)
    image_buffer = metric_utils.buffer_plot(plot)
    return image_buffer
