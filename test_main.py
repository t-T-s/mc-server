import pytest
from fastapi.testclient import TestClient
from main import app

# Create a TestClient to make requests to your FastAPI application
client = TestClient(app)


def test_post_accuracy_1d():
    # Inputs as 1d arrays for accuracy check
    item_data = {"ground_truth": [1, 2, 2, 4, 5],
                 "predictions": [1, 2, 3, 4, 5]}
    response = client.post("/clf_accuracy", json=item_data)
    assert response.status_code == 200
    assert response.json() == {"accuracy": 0.80}


def test_post_accuracy_label_indicator():
    # Inputs as label indicator arrays. There can be multiple columns
    item_data = {"ground_truth": [[0, 1], [1, 0]],
                 "predictions": [[0, 1], [1, 1]]}
    response = client.post("/clf_accuracy", json=item_data)
    assert response.status_code == 200
    assert response.json() == {"accuracy": 0.50}


def test_post_accuracy_label_indicator_inner_list_lengths():
    # Inputs as label indicator arrays of different nested list lengths.
    item_data = {"ground_truth": [[0, 1], [1, 1], [1]],
                 "predictions": [[0, 1], [1, 1], [0]]}
    response = client.post("/clf_accuracy", json=item_data)
    assert response.status_code != 200


def test_post_accuracy_1d_length():
    # Inputs as 1d arrays for accuracy check
    item_data = {"ground_truth": [1, 2, 2, 4, 5, 8],
                 "predictions": [1, 2, 3, 4, 5]}
    response = client.post("/clf_accuracy", json=item_data)
    assert response.status_code != 200


def test_post_consistency_scores():
    # Basic invocation test
    contrib_dict = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                    'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                    'LIME': [[0.14, 0.2], [0.34, 0.4]]}
    response = client.post("/consistency", json=contrib_dict)
    assert response.status_code == 200
    assert response.json().keys() == {"average_consistency", "pairwise_scores"}


if __name__ == '__main__':
    pytest.main(['-sv', __file__])
