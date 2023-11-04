from fastapi import FastAPI, HTTPException
from metrics import clf_accuracy_score, consistency_scores, compacity_scores, impact_score
from data_validators import ClfLabels, Contributions

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Root"}


@app.post("/clf_accuracy", status_code=200)
async def get_accuracy(payload: ClfLabels):
    ground_truth = payload.ground_truth
    predictions = payload.predictions

    metric_result = clf_accuracy_score(ground_truth, predictions)

    response_model = {"accuracy": metric_result}
    # TODO: Validate the response with pydantic
    return response_model


@app.post("/consistency", status_code=200)
async def get_consistency(payload: Contributions):
    contributions = payload.contribution_dict

    average_consistency, pairwise_scores = consistency_scores(contributions)

    response_model = {"average_consistency": average_consistency,
                      "pairwise_scores": pairwise_scores}
    # TODO: Validate the response with pydantic
    return response_model
