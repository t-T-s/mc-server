from app.core.metrics import scikit_accuracy, shapash_consistency, shapash_compacity_from_contributions, \
    evasion_impact, shapash_consistency_plot, shapash_compacity_plot, tsne_user_diversity, tsne_user_diversity_plot, \
    shapash_stability_plot
from app.core.schemas.schema import StabilityData
from app.core import metric_utils
import joblib, io
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


def stability_surrogate_model_plot(x_test, contributions, y_target, selection, max_points, max_features, x_train=None
                                   , y_train=None):
    ## check if the x_train and y_train are not none
    if x_train is None or y_train is None:
        raise ValueError("The training data x_train and y_train must be provided to train the surrogate model")
    stability_graph = shapash_stability_plot(
        x_train=x_train
        , y_train=y_train
        , x_test=x_test
        , contributions=contributions
        , y_target=y_target
        , selection=selection
        , max_points=max_points
        , max_features=max_features)
    image_buffer = metric_utils.buffer_plot(stability_graph)
    return image_buffer


def stability_pre_trained_model_plot(file_bytes):

    data_dict = joblib.load(io.BytesIO(file_bytes))
    payload = StabilityData(**data_dict)
    # check if the received model is not none
    if payload.pre_trained_model is None:
        raise ValueError("The received model is not provided. Please provide the model to plot the stability")

    contributions = payload.contributions
    x_input = payload.x_input
    y_target = payload.y_target
    selection = payload.selection
    max_points = payload.max_points
    max_features = payload.max_features

    model: BaseEstimator = joblib.load(io.BytesIO(payload.pre_trained_model))  # type hinting only
    if not isinstance(model, BaseEstimator):
        raise TypeError("Currently the stability plot with an externally trained model is only supported for sci-kit "
                        "learn models."
                        "The model uploaded is not of type sklearn.base.BaseEstimator. "
                        "You can opt for training a surrogate model to plot the stability")
    stability_graph = shapash_stability_plot(
        model=model
        , x_test=x_input
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
