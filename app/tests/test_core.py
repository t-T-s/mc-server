import io

import pandas
import pytest
from app.core import metrics
from app.core import metric_utils
from shapash.explainer.consistency import Consistency
import matplotlib.pyplot as plt

from app.core.metrics import tsne_user_diversity


def test_shapash_compacity():
    # Output test: Basic invocation test
    contributions = [[0.15, 0.2, 0.4, 0.01], [0.3, 0.42, 0.34, 0.012], [0.3, 0.42, 0.24, 0.011]]
    selection = [0, 1, 2]
    distance = 0.9
    nb_features = 2
    pd_contributions = metric_utils.convert_contrib_to_dataframe(contributions)
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
    consistency_plot.savefig('assets/test_consistency_core_image.png', bbox_inches='tight')
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
