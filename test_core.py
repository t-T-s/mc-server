import pytest
from core import shapash_compacity_from_contributions
from mc_utils import convert_contrib_to_dataframe, create_dummy_model
import numpy as np


def test_shapash_compacity():
    # Output test: Basic invocation test
    contributions = [[0.15, 0.2, 0.4, 0.01], [0.3, 0.42, 0.34, 0.012], [0.3, 0.42, 0.24, 0.011]]
    selection = [0, 1, 2]
    distance = 0.9
    nb_features = 2
    pd_contribs = convert_contrib_to_dataframe(contributions)
    feature_compacity = shapash_compacity_from_contributions(contributions=pd_contribs
                                          , selection=selection
                                          , distance=distance
                                          , nb_features=nb_features)
    print(feature_compacity)
    assert 'features_needed' in feature_compacity.keys()
    assert 'distance_reached' in feature_compacity.keys()
    assert isinstance(feature_compacity['features_needed'], list)
    assert isinstance(feature_compacity['distance_reached'], list)


if __name__ == '__main__':
    pytest.main(['-sv', __file__])
