import pytest
from fastapi.testclient import TestClient
from app.main import app
from typing import Dict

# Create a TestClient to make requests to your FastAPI application
client = TestClient(app)


def test_post_accuracy_1d():
    # Input test: as 1d arrays for accuracy check
    item_data = {"ground_truth": [1, 2, 2, 4, 5],
                 "predictions": [1, 2, 3, 4, 5]}
    response = client.post("/clf_accuracy_metric", json=item_data)
    assert response.status_code == 200
    assert isinstance(response.json()['accuracy'], float)


def test_post_accuracy_label_indicator():
    # Input test: as label indicator arrays. There can be multiple columns
    item_data = {"ground_truth": [[0, 1], [1, 0]],
                 "predictions": [[0, 1], [1, 1]]}
    response = client.post("/clf_accuracy_metric", json=item_data)
    assert response.status_code == 200
    assert response.json() == {"accuracy": 0.50}


def test_post_accuracy_label_indicator_inner_list_lengths():
    # Input test: as label indicator arrays of different nested list lengths.
    item_data = {"ground_truth": [[0, 1], [1, 1], [1]],
                 "predictions": [[0, 1], [1, 1], [0]]}
    response = client.post("/clf_accuracy_metric", json=item_data)
    assert response.status_code != 200


def test_post_accuracy_1d_length():
    # Input test: as 1d arrays with different lengths
    item_data = {"ground_truth": [1, 2, 2, 4, 5, 8],
                 "predictions": [1, 2, 3, 4, 5]}
    response = client.post("/clf_accuracy_metric", json=item_data)
    assert response.status_code != 200


def test_post_consistency_scores():
    # Output test: Basic invocation test
    contrib_dict = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                    'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                    'LIME': [[0.14, 0.2], [0.34, 0.4]]}
    response = client.post("/consistency_metric", json=contrib_dict)
    print(response.json())
    assert response.status_code == 200
    assert 'average_consistency' in response.json().keys()
    assert 'pairwise_scores' in response.json().keys()
    assert isinstance(response.json()["average_consistency"], float)
    assert isinstance(response.json()["pairwise_scores"], Dict)


def test_post_consistency_plot():
    # Output test: Basic invocation test
    contrib_dict = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                    'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                    'LIME': [[0.14, 0.2], [0.34, 0.4]]}
    response = client.post("/consistency_metric_plot", json=contrib_dict)
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'image/png'
    with open("assets/test_consistency_endpoint_response_image.png", "wb") as f:
        f.write(response.content)
    print(response.headers)
    print("Image saved to test_consistency_endpoint_response_image.png successfully!")


def test_post_compacity_plot():
    # Output test: Basic invocation test
    item_data = {"contributions": [[0.15, 0.2, 0.4, 0.01], [0.3, 0.42, 0.34, 0.012]],
                 "selection": [0, 1],
                 "distance": 0.9,
                 "nb_features": 2}
    response = client.post("/compacity_metric_plot", json=item_data)
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'image/png'
    with open("assets/test_compacity_endpoint_response_image.png", "wb") as f:
        f.write(response.content)
    print(response.headers)
    print("Image saved to test_compacity_endpoint_response_image.png successfully!")


def test_post_compacity_contributions_row_lengths():
    # Input test: as label indicator arrays of different nested list lengths.
    item_data = {"contributions": [[0.15, 0.2, 0.4, 0.01], [0.3, 0.42, 0.34, 0.012]],
                 "selection": [0, 1],
                 "distance": 0.9,
                 "nb_features": 2}
    response = client.post("/compacity_metric", json=item_data)
    print(response.json())
    assert response.status_code == 200


def test_post_impact_test():
    # Input test: as 1d arrays for impact test
    item_data = {"ground_truth": [1, 2, 2, 4, 5],
                 "predictions": [1, 2, 3, 4, 5]}
    response = client.post("/evasion_impact_metric", json=item_data)
    assert response.status_code == 200
    assert isinstance(response.json()['impact'], float)


if __name__ == '__main__':
    pytest.main(['-sv', __file__])
