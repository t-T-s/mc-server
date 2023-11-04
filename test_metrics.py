import pytest
from metrics import consistency_scores


def test_consistency_scores():
    # Basic invocation test
    contrib_dict = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                    'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                    'LIME': [[0.14, 0.2], [0.34, 0.4]]}
    avg, pw_cons = consistency_scores(contributions=contrib_dict)
    assert avg is not None
    assert pw_cons is not None


if __name__ == '__main__':
    pytest.main(['-sv', __file__])
