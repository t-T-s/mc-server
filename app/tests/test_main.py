import pytest
from fastapi.testclient import TestClient
from app.main import app
from typing import Dict
from app.core import metric_utils
import io
import joblib
import json

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


def test_post_stability_with_surrogate_model_training_plot():
    # Output test: Basic invocation test
    item_data = {
        "x_train": [[0.24763825, 0.4624466, 0.1439733, 0.63356432],
                    [0.89960405, 0.60607923, 0.58054955, 0.07378852],
                    [0.91335706, 0.77419346, 0.70098694, 0.08870475],
                    [0.46562798, 0.85730786, 0.53299792, 0.84305255],
                    [0.32298965, 0.86707571, 0.73935329, 0.8347728]],
        "y_train": [[1], [1], [1], [1], [0]],
        "x_input": [[0.46582937, 0.36313128, 0.17189367, 0.01546506],
                    [0.507121, 0.01918931, 0.78464877, 0.77427306],
                    [0.50496966, 0.98061098, 0.27967449, 0.29417485],
                    [0.3521993, 0.48835738, 0.23460793, 0.64657831],
                    [0.67447758, 0.39336409, 0.01272322, 0.06723874],
                    [0.64747419, 0.46749566, 0.35986405, 0.95362188],
                    [0.68757052, 0.66098619, 0.99523119, 0.09020147],
                    [0.50099671, 0.93649314, 0.25915279, 0.75566948],
                    [0.87476536, 0.04664153, 0.89152254, 0.49654976],
                    [0.87059629, 0.86237521, 0.85991116, 0.08158515],
                    [0.70103771, 0.28800044, 0.20746705, 0.44251794]],
        "contributions": [[0.40302247, 0.2380319, 0.2301524, 0.51886267],
                          [0.67702666, 0.56105417, 0.2590706, 0.16573209],
                          [0.89889708, 0.20274018, 0.35884345, 0.81657564],
                          [0.90689996, 0.47364288, 0.7311526, 0.62089024],
                          [0.40710054, 0.14887248, 0.84189794, 0.71489193],
                          [0.08248856, 0.35354858, 0.11228026, 0.99185406],
                          [0.16592895, 0.53207895, 0.33886526, 0.5177407],
                          [0.81840455, 0.52111217, 0.81772124, 0.15083058],
                          [0.84590188, 0.52619182, 0.89583799, 0.21780331],
                          [0.09295941, 0.83894879, 0.46052668, 0.1308371],
                          [0.07235874, 0.92571017, 0.05129698, 0.92341386]],
        "y_target": [[1], [1], [0], [1], [0], [0], [0], [1], [0], [1], [1]],
        "selection": [0, 1],
        "max_points": 500,
        "max_features": 2
    }

    response = client.post(
        "/stability_surrogate_model_metric_plot", json=item_data
    )
    print(response)
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'image/png'
    with open("assets/test_post_stability_with_surrogate_model_training_plot_endpoint_response_image.png", "wb") as f:
        f.write(response.content)
    print(response.headers)
    print(
        "Image saved to test_post_stability_with_surrogate_model_training_plot_endpoint_response_image.png successfully!")


def test_post_stability_with_pretrained_model_plot():
    rf_model = metric_utils.create_dummy_model()
    model_buf = io.BytesIO()
    joblib.dump(rf_model, model_buf)
    # Output test: Basic invocation test
    item_data = {
        "x_input": [[0.46582937, 0.36313128, 0.17189367, 0.01546506],
                    [0.507121, 0.01918931, 0.78464877, 0.77427306],
                    [0.50496966, 0.98061098, 0.27967449, 0.29417485],
                    [0.3521993, 0.48835738, 0.23460793, 0.64657831],
                    [0.67447758, 0.39336409, 0.01272322, 0.06723874],
                    [0.64747419, 0.46749566, 0.35986405, 0.95362188],
                    [0.68757052, 0.66098619, 0.99523119, 0.09020147],
                    [0.50099671, 0.93649314, 0.25915279, 0.75566948],
                    [0.87476536, 0.04664153, 0.89152254, 0.49654976],
                    [0.87059629, 0.86237521, 0.85991116, 0.08158515],
                    [0.70103771, 0.28800044, 0.20746705, 0.44251794]],
        "contributions": [[0.40302247, 0.2380319, 0.2301524, 0.51886267],
                          [0.67702666, 0.56105417, 0.2590706, 0.16573209],
                          [0.89889708, 0.20274018, 0.35884345, 0.81657564],
                          [0.90689996, 0.47364288, 0.7311526, 0.62089024],
                          [0.40710054, 0.14887248, 0.84189794, 0.71489193],
                          [0.08248856, 0.35354858, 0.11228026, 0.99185406],
                          [0.16592895, 0.53207895, 0.33886526, 0.5177407],
                          [0.81840455, 0.52111217, 0.81772124, 0.15083058],
                          [0.84590188, 0.52619182, 0.89583799, 0.21780331],
                          [0.09295941, 0.83894879, 0.46052668, 0.1308371],
                          [0.07235874, 0.92571017, 0.05129698, 0.92341386]],
        "y_target": [[1], [1], [0], [1], [0], [0], [0], [1], [0], [1], [1]],
        "selection": [0, 1],
        "max_points": 500,
        "max_features": 2,
        "pre_trained_model": model_buf.getvalue()
    }
    # item_data = {"Hey": "hey"}
    request_buf = io.BytesIO()
    joblib.dump(item_data, request_buf)

    response = client.post(
        "/stability_pre_trained_model_metric_plot",
        files={"file": ("request.bytes", bytes(request_buf.getvalue()))}
    )
    print(response)
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'image/png'
    with open("assets/test_post_stability_with_pretrained_model_plot_endpoint_response_image.png", "wb") as f:
        f.write(response.content)
    print(response.headers)
    print("Image saved to test_post_stability_with_pretrained_model_plot_endpoint_response_image.png successfully!")


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


def test_post_user_diversity_plot():
    # Output test: Basic invocation test
    item_data = {
        "predictions": [[-2.218350887298584, -2.198277711868286],
                        [-2.5687193870544434, -2.458390474319458],
                        [-2.0745654106140137, -2.329625368118286],
                        [-2.2383768558502197, -2.222764253616333],
                        [-2.269338846206665, -2.456698179244995],
                        [-1.9543006420135498, -2.549536943435669]],
        "client_ids": [1, 1, 1, 2, 2, 2],
        "perplexity": 4
    }
    response = client.post("/user_diversity_metric_plot", json=item_data)
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'image/png'
    with open("assets/test_user_diversity_image.png", "wb") as f:
        f.write(response.content)
    print(response.headers)
    print("Image saved to test_user_diversity_image.png successfully!")


if __name__ == '__main__':
    pytest.main(['-sv', __file__])
