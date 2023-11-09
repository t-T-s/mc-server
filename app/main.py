from fastapi import FastAPI
from app.service.metrics_service import clf_accuracy_score, consistency_scores, compacity_scores, evasion_impact_score
from app.core.schemas.schema import ClfLabels, ContributionsDict, Contributions

app = FastAPI()


@app.post("/clf_accuracy_metric", status_code=200)
async def post_accuracy(payload: ClfLabels):
    ground_truth = payload.ground_truth
    predictions = payload.predictions

    metric_result = clf_accuracy_score(ground_truth, predictions)

    response_model = {"accuracy": metric_result}
    # TODO: Validate the response with pydantic
    return response_model


@app.post("/consistency_metric", status_code=200)
async def post_consistency(payload: ContributionsDict):
    contributions = payload.contribution_dict

    average_consistency, pairwise_scores = consistency_scores(contributions)

    response_model = {"average_consistency": average_consistency,
                      "pairwise_scores": pairwise_scores}
    # TODO: Validate the response with pydantic
    return response_model


# To be implemented
@app.post("/compacity_metric", status_code=200)
async def post_compacity(payload: Contributions):
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


# To be implemented
@app.post("/evasion_impact_metric", status_code=200)
async def post_evasion_impact(payload: ClfLabels):
    ground_truth = payload.ground_truth
    predictions = payload.predictions

    metric_result = evasion_impact_score(ground_truth=ground_truth
                                         , predictions=predictions)
    response_model = {"impact": metric_result}
    # TODO: Validate the response with pydantic
    return response_model
