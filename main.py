from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from metrics import clf_accuracy_score
from typing import List

app = FastAPI()

# pydantic models


class ClfLabels(BaseModel):
    ground_truth: List[int] = [1, 2, 3, 4, 5]
    predictions: List[int] = [1, 2, 3, 4, 5]


class MetricResponse(BaseModel):
    metric_value: dict


@app.get("/")
async def root():
    return {"message": "Root"}

#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}


@app.post("/clf_accuracy", status_code=200)
async def get_accuracy(payload: ClfLabels):
    ground_truth = payload.ground_truth
    predictions = payload.predictions

    metric_result = clf_accuracy_score(ground_truth, predictions)

    if not metric_result:
        raise HTTPException(status_code=400, detail="Something went wrong")
    if metric_result == 0:
        raise HTTPException(status_code=406, detail="Input lengths are different")
    response_model = {"accuracy": metric_result}

    return response_model
