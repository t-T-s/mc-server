import pytest
from fastapi.testclient import TestClient
from main import app

# Create a TestClient to make requests to your FastAPI application
client = TestClient(app)


def test_get_accuracy_1d():
    # Inputs as 1d arrays for accuracy check
    item_data = {"ground_truth": [1, 2, 2, 4, 5],
                 "predictions": [1, 2, 3, 4, 5]}
    response = client.post("/clf_accuracy", json=item_data)
    assert response.status_code == 200
    assert response.json() == {"accuracy": 0.80}


def test_get_accuracy_label_indicator():
    # Inputs as label indicator arrays. There can be multiple columns
    item_data = {"ground_truth": [[0, 1], [1, 1]],
                 "predictions": [[0, 1], [1, 1]]}
    response = client.post("/clf_accuracy", json=item_data)
    assert response.status_code == 200
    assert response.json() == {"accuracy": 0.50}


def test_get_accuracy_label_indicator_lengths():
    # Inputs as label indicator arrays of different nested list lengths.
    item_data = {"ground_truth": [[0, 1], [1, 1], [1]],
                 "predictions": [[0, 1], [1, 1], [0]]}
    response = client.post("/clf_accuracy", json=item_data)
    assert response.status_code != 200

