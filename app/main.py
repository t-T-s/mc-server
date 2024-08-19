import json

from fastapi import FastAPI, Response, BackgroundTasks, UploadFile, File
from app.services.metrics_service import clf_accuracy_score, consistency_scores, compacity_scores, \
    evasion_impact_score, consistency_plot, compacity_plot, user_diversity_score, user_diversity_plot, \
    stability_surrogate_model_plot, stability_pre_trained_model_plot
from app.core.schemas.schema import ClfLabels, ConsistencyContributions, CompacityContributions, UserDiversityInput, \
    StabilityData
from app.utils.mc_exceptions import http_exception_handler, global_exception_handler
from fastapi import HTTPException

app = FastAPI()
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)


@app.post("/clf_accuracy_metric", status_code=200)
async def post_accuracy(payload: ClfLabels):
    ground_truth = payload.ground_truth
    predictions = payload.predictions

    metric_result = clf_accuracy_score(ground_truth, predictions)

    response_model = {"accuracy": metric_result}
    # TODO: Validate the response with pydantic
    return response_model


@app.post("/consistency_metric", status_code=200)
async def post_consistency(payload: ConsistencyContributions):
    contributions = payload.contribution_dict

    average_consistency, pairwise_scores = consistency_scores(contributions)

    response_model = {"average_consistency": average_consistency,
                      "pairwise_scores": pairwise_scores}
    # TODO: Validate the response with pydantic
    return response_model


@app.post("/consistency_metric_plot", status_code=200)
async def post_consistency_plot(payload: ConsistencyContributions
                                , background_tasks: BackgroundTasks):
    contributions = payload.contribution_dict

    image_buffer = consistency_plot(contributions)
    buf_contents: bytes = image_buffer.getvalue()
    background_tasks.add_task(image_buffer.close)
    headers = {'Content-Disposition': 'inline; filename="consistency.png"'}
    return Response(buf_contents, headers=headers, media_type='image/png')


# To be implemented
@app.post("/compacity_metric", status_code=200)
async def post_compacity(payload: CompacityContributions):
    contributions = payload.contributions
    selection = payload.selection
    distance = payload.distance
    nb_features = payload.nb_features

    response_model = compacity_scores(contributions=contributions
                                      , selection=selection
                                      , distance=distance
                                      , nb_features=nb_features)
    # TODO: Validate the response with pydantic
    return response_model


@app.post("/compacity_metric_plot", status_code=200)
async def post_compacity_plot(payload: CompacityContributions
                              , background_tasks: BackgroundTasks):
    contributions = payload.contributions
    selection = payload.selection
    distance = payload.distance
    nb_features = payload.nb_features

    image_buffer = compacity_plot(contributions=contributions
                                  , selection=selection
                                  , distance=distance
                                  , nb_features=nb_features)
    buf_contents: bytes = image_buffer.getvalue()
    background_tasks.add_task(image_buffer.close)
    headers = {'Content-Disposition': 'inline; filename="compacity.png"'}
    return Response(buf_contents, headers=headers, media_type='image/png')


@app.post("/stability_surrogate_model_metric_plot", status_code=200)
async def post_stability_with_surrogate_model_training_plot(background_tasks: BackgroundTasks
                                                            , payload: StabilityData
                                                            ):
    """
     The stability plot is generated using this endpoint. The endpoint receives a json with the following keys.
     (refer the test_post_stability_with_surrogate_model_training_plot for more implementation details).
    :param StabilityData payload:
    :return Response response: Image of the stability plot.
    """

    x_train = payload.x_train
    y_train = payload.y_train
    contributions = payload.contributions
    x_input = payload.x_input
    y_target = payload.y_target
    selection = payload.selection
    max_points = payload.max_points
    max_features = payload.max_features

    image_buffer = stability_surrogate_model_plot(
        x_test=x_input
        , contributions=contributions
        , y_target=y_target
        , selection=selection
        , max_points=max_points
        , max_features=max_features
        , x_train=x_train
        , y_train=y_train
    )
    buf_contents: bytes = image_buffer.getvalue()
    background_tasks.add_task(image_buffer.close)
    headers = {'Content-Disposition': 'inline; filename="stability_with_surrogate_model_training.png"',
               'Stability-calculation-type': 'Using surrogate model'
               }
    return Response(buf_contents, headers=headers, media_type='image/png')


@app.post("/stability_pre_trained_model_metric_plot", status_code=200
    , summary="Generate the stability plot with a pre-trained model")
async def post_stability_with_pre_trained_model_plot(background_tasks: BackgroundTasks,
                                                     file: UploadFile = File(...)):
    """
     Generates the stability plot. The endpoint should receive a file preferably a dictionary
     (refer the test_post_stability_with_pre_trained_model_plot for more implementation details).

    - **param** bytes file: The file should contain fields of StabilityFileData schema as a dictionary.
                            The following are the keys of the dictionary that should be sent as a file:
        - **x_input**: List[List[float]]\*: The input data for the model.
        - **contributions** List[List[float]]\*: The contributions of the model.
        - **y_target** List[List[float]]\*: The target data for the model.
        - **selection** Union[List[int], None]: The selected features.
        - **max_points** Union[int, None]: default=500 The maximum number of points for neighbourhood.
        - **max_features** Union[int, None]: The maximum number of features use.
        - **pre_trained_model** bytes: The pre-trained model.
    - **return** .PNG stability_with_pre_trained_model.png: Image of the stability plot.
    """
    file_bytes = await file.read()
    image_buffer = stability_pre_trained_model_plot(file_bytes)
    buf_contents: bytes = image_buffer.getvalue()
    background_tasks.add_task(image_buffer.close)
    headers = {'Content-Disposition': 'inline; filename="stability_with_pre_trained_model.png"',
               'Stability-calculation-type': 'Using the pre-trained model'
               }
    return Response(buf_contents, headers=headers, media_type='image/png')


@app.post("/evasion_impact_metric", status_code=200)
async def post_evasion_impact(payload: ClfLabels):
    ground_truth = payload.ground_truth
    predictions = payload.predictions

    metric_result = evasion_impact_score(ground_truth=ground_truth
                                         , predictions=predictions)
    response_model = {"impact": metric_result}
    # TODO: Validate the response with pydantic
    return response_model


@app.post("/user_diversity_metric", status_code=200)
async def post_user_diversity(payload: UserDiversityInput):
    x_results = payload.predictions
    y_results = payload.client_ids
    perplexity = payload.perplexity

    response_model = user_diversity_score(x_results, y_results, perplexity)
    # TODO: Validate the response with pydantic
    return response_model


@app.post("/user_diversity_metric_plot", status_code=200)
async def post_user_diversity(payload: UserDiversityInput, background_tasks: BackgroundTasks):
    x_results = payload.predictions
    y_results = payload.client_ids
    perplexity = payload.perplexity

    image_buffer = user_diversity_plot(x_results, y_results, perplexity)
    buf_contents: bytes = image_buffer.getvalue()
    background_tasks.add_task(image_buffer.close)
    headers = {'Content-Disposition': 'inline; filename="compacity.png"'}
    return Response(buf_contents, headers=headers, media_type='image/png')
