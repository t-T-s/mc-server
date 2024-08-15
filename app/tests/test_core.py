import io

import pandas
import pytest
from app.core import metrics
from app.core import metric_utils
from shapash.explainer.consistency import Consistency
import matplotlib.pyplot as plt
import joblib

from app.core.metrics import tsne_user_diversity


def test_shapash_compacity():
    # Output test: Basic invocation test
    contributions = [[0.15, 0.2, 0.4, 0.01], [0.3, 0.42, 0.34, 0.012], [0.3, 0.42, 0.24, 0.011]]
    selection = [0, 1, 2]
    distance = 0.9
    nb_features = 2
    pd_contributions = metric_utils.convert_array_like_to_dataframe(contributions)
    feature_compacity = metrics.shapash_compacity_from_contributions(contributions=pd_contributions
                                                                     , selection=selection
                                                                     , distance=distance
                                                                     , nb_features=nb_features)
    print(feature_compacity)
    assert 'features_needed' in feature_compacity.keys()
    assert 'distance_reached' in feature_compacity.keys()
    assert isinstance(feature_compacity['features_needed'], list)
    assert isinstance(feature_compacity['distance_reached'], list)


def test_plot_consistency():
    # Test the output of metric_utils.plot_consistency
    contrib_dict = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                    'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                    'LIME': [[0.14, 0.2], [0.34, 0.4]]}
    cns = Consistency()
    pd_contributions = metric_utils.convert_contrib_dict_to_dataframe(contrib_dict)
    cns.compile(contributions=pd_contributions)
    consistency_plot = metric_utils.plot_consistency(cns)
    # consistency_plot.savefig('assets/test_consistency_core_image.png', bbox_inches='tight')
    assert isinstance(consistency_plot, plt.Figure)


def test_shapash_compacity_plot():
    # Test the output of metrics.shapash_compacity_plot
    contributions = [[0.15, 0.2, 0.4, 0.01], [0.3, 0.42, 0.34, 0.012], [0.3, 0.42, 0.24, 0.011]]
    selection = [0, 1, 2]
    distance = 0.9
    nb_features = 2
    compacity_plot = metrics.shapash_compacity_plot(contributions=contributions
                                                    , selection=selection
                                                    , distance=distance
                                                    , nb_features=nb_features)
    img_buf = io.BytesIO()
    img_bytes = compacity_plot.to_image('png', engine='kaleido')
    compacity_plot.write_image("assets/test_compacity_core_image.png", format="png")
    compacity_plot.write_image(img_buf, format="png")
    assert isinstance(img_buf, io.BytesIO)
    assert isinstance(img_bytes, bytes)


def test_shapash_stability_plot():
    # Test the output of metrics.shapash_compacity_plot
    # rf_model = metric_utils.create_dummy_model()
    # model_file_name = '"assets/dummy_model.joblib"'
    # joblib.dump(rf_model, model_file_name)
    # loaded_model = joblib.load(model_file_name)
    selection = None
    max_points = 500
    max_features = 2
    # The number of inputs have to be greater than 10
    x = [[0.46582937, 0.36313128, 0.17189367, 0.01546506],
         [0.507121, 0.01918931, 0.78464877, 0.77427306],
         [0.50496966, 0.98061098, 0.27967449, 0.29417485],
         [0.3521993, 0.48835738, 0.23460793, 0.64657831],
         [0.67447758, 0.39336409, 0.01272322, 0.06723874],
         [0.64747419, 0.46749566, 0.35986405, 0.95362188],
         [0.68757052, 0.66098619, 0.99523119, 0.09020147],
         [0.50099671, 0.93649314, 0.25915279, 0.75566948],
         [0.87476536, 0.04664153, 0.89152254, 0.49654976],
         [0.87059629, 0.86237521, 0.85991116, 0.08158515],
         [0.70103771, 0.28800044, 0.20746705, 0.44251794]]
    contributions = [[0.40302247, 0.2380319, 0.2301524, 0.51886267],
                     [0.67702666, 0.56105417, 0.2590706, 0.16573209],
                     [0.89889708, 0.20274018, 0.35884345, 0.81657564],
                     [0.90689996, 0.47364288, 0.7311526, 0.62089024],
                     [0.40710054, 0.14887248, 0.84189794, 0.71489193],
                     [0.08248856, 0.35354858, 0.11228026, 0.99185406],
                     [0.16592895, 0.53207895, 0.33886526, 0.5177407],
                     [0.81840455, 0.52111217, 0.81772124, 0.15083058],
                     [0.84590188, 0.52619182, 0.89583799, 0.21780331],
                     [0.09295941, 0.83894879, 0.46052668, 0.1308371],
                     [0.07235874, 0.92571017, 0.05129698, 0.92341386]]
    y_target = [[1], [1], [0], [1], [0], [0], [0], [1], [0], [1], [1]]
    stability_plot = metrics.shapash_stability_plot(
        model=None
        , x_encoded=x
        , contributions=contributions
        , y_target=y_target
        , selection=selection
        , max_points=max_points
        , max_features=max_features)
    img_buf = io.BytesIO()
    img_bytes = stability_plot.to_image('png', engine='kaleido')
    stability_plot.write_image("assets/test_stability_core_image.png", format="png")
    stability_plot.write_image(img_buf, format="png")
    assert isinstance(img_buf, io.BytesIO)
    assert isinstance(img_bytes, bytes)

def test_shapash_stability_with_surrogate_model_training_plot():
    # Test the output of metrics.shapash_compacity_plot
    # rf_model = metric_utils.create_dummy_model()
    # model_file_name = '"assets/dummy_model.joblib"'
    # joblib.dump(rf_model, model_file_name)
    # loaded_model = joblib.load(model_file_name)
    selection = None
    max_points = 500
    max_features = 2
    x_train=[[0.24763825, 0.4624466 , 0.1439733 , 0.63356432],
       [0.89960405, 0.60607923, 0.58054955, 0.07378852],
       [0.91335706, 0.77419346, 0.70098694, 0.08870475],
       [0.46562798, 0.85730786, 0.53299792, 0.84305255],
       [0.32298965, 0.86707571, 0.73935329, 0.8347728 ]]
    y_train= [[1], [1], [1], [1], [0]]
    # The number of test inputs have to be greater than 10
    x = [[0.46582937, 0.36313128, 0.17189367, 0.01546506],
         [0.507121, 0.01918931, 0.78464877, 0.77427306],
         [0.50496966, 0.98061098, 0.27967449, 0.29417485],
         [0.3521993, 0.48835738, 0.23460793, 0.64657831],
         [0.67447758, 0.39336409, 0.01272322, 0.06723874],
         [0.64747419, 0.46749566, 0.35986405, 0.95362188],
         [0.68757052, 0.66098619, 0.99523119, 0.09020147],
         [0.50099671, 0.93649314, 0.25915279, 0.75566948],
         [0.87476536, 0.04664153, 0.89152254, 0.49654976],
         [0.87059629, 0.86237521, 0.85991116, 0.08158515],
         [0.70103771, 0.28800044, 0.20746705, 0.44251794]]
    contributions = [[0.40302247, 0.2380319, 0.2301524, 0.51886267],
                     [0.67702666, 0.56105417, 0.2590706, 0.16573209],
                     [0.89889708, 0.20274018, 0.35884345, 0.81657564],
                     [0.90689996, 0.47364288, 0.7311526, 0.62089024],
                     [0.40710054, 0.14887248, 0.84189794, 0.71489193],
                     [0.08248856, 0.35354858, 0.11228026, 0.99185406],
                     [0.16592895, 0.53207895, 0.33886526, 0.5177407],
                     [0.81840455, 0.52111217, 0.81772124, 0.15083058],
                     [0.84590188, 0.52619182, 0.89583799, 0.21780331],
                     [0.09295941, 0.83894879, 0.46052668, 0.1308371],
                     [0.07235874, 0.92571017, 0.05129698, 0.92341386]]
    y_target = [[1], [1], [0], [1], [0], [0], [0], [1], [0], [1], [1]]
    stability_plot = metrics.shapash_stability_plot(
          x_train=x_train
        , y_train=y_train
        , model=None
        , x_test=x
        , contributions=contributions
        , y_target=y_target
        , selection=selection
        , max_points=max_points
        , max_features=max_features)
    img_buf = io.BytesIO()
    img_bytes = stability_plot.to_image('png', engine='kaleido')
    stability_plot.write_image("assets/test_stability_core_image.png", format="png")
    stability_plot.write_image(img_buf, format="png")
    assert isinstance(img_buf, io.BytesIO)
    assert isinstance(img_bytes, bytes)


def test_shapash_stability():
    # Test the output of metrics.shapash_compacity_plot
    selection = None
    max_points = 500
    max_features = 2
    # The number of inputs have to be greater than 10
    x = [[0.46582937, 0.36313128, 0.17189367, 0.01546506],
         [0.507121, 0.01918931, 0.78464877, 0.77427306],
         [0.50496966, 0.98061098, 0.27967449, 0.29417485],
         [0.3521993, 0.48835738, 0.23460793, 0.64657831],
         [0.67447758, 0.39336409, 0.01272322, 0.06723874],
         [0.64747419, 0.46749566, 0.35986405, 0.95362188],
         [0.68757052, 0.66098619, 0.99523119, 0.09020147],
         [0.50099671, 0.93649314, 0.25915279, 0.75566948],
         [0.87476536, 0.04664153, 0.89152254, 0.49654976],
         [0.87059629, 0.86237521, 0.85991116, 0.08158515],
         [0.70103771, 0.28800044, 0.20746705, 0.44251794]]
    contributions = [[0.40302247, 0.2380319, 0.2301524, 0.51886267],
                     [0.67702666, 0.56105417, 0.2590706, 0.16573209],
                     [0.89889708, 0.20274018, 0.35884345, 0.81657564],
                     [0.90689996, 0.47364288, 0.7311526, 0.62089024],
                     [0.40710054, 0.14887248, 0.84189794, 0.71489193],
                     [0.08248856, 0.35354858, 0.11228026, 0.99185406],
                     [0.16592895, 0.53207895, 0.33886526, 0.5177407],
                     [0.81840455, 0.52111217, 0.81772124, 0.15083058],
                     [0.84590188, 0.52619182, 0.89583799, 0.21780331],
                     [0.09295941, 0.83894879, 0.46052668, 0.1308371],
                     [0.07235874, 0.92571017, 0.05129698, 0.92341386]]
    y_target = [[1], [1], [0], [1], [0], [0], [0], [1], [0], [1], [1]]
    features_stability = metrics.shapash_feature_stability(x_encoded=x
                                                           , contributions=contributions
                                                           , y_target=y_target
                                                           , selection=selection
                                                           )
    print(features_stability)
    assert 'features_needed' in features_stability.keys()
    assert 'distance_reached' in features_stability.keys()
    assert isinstance(features_stability['variability'], list)
    assert isinstance(features_stability['amplitude'], list)

def test_user_diversity_metric():
    predictions = [[-2.218350887298584, -2.198277711868286],
                   [-2.5687193870544434, -2.458390474319458],
                   [-2.0745654106140137, -2.329625368118286],
                   [-2.2383768558502197, -2.222764253616333],
                   [-2.269338846206665, -2.456698179244995],
                   [-1.9543006420135498, -2.549536943435669]]
    client_ids = [1, 1, 1, 2, 2, 2]
    perplexity = 4
    diversity = tsne_user_diversity(predictions, client_ids, perplexity)
    assert isinstance(diversity[0], str)


if __name__ == '__main__':
    pytest.main(['-sv', __file__])
