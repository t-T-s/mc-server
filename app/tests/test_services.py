import pytest
from app.services.metrics_service import consistency_scores, consistency_plot
import io


def test_consistency_scores():
    # Basic invocation test
    contrib_dict = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                    'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                    'LIME': [[0.14, 0.2], [0.34, 0.4]]}
    avg, pw_cons = consistency_scores(contributions_dict=contrib_dict)
    print(avg, pw_cons)
    assert isinstance(avg, float)
    assert isinstance(pw_cons, dict)


def test_consistency_plot():
    # Basic invocation test
    contrib_dict = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                    'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                    'LIME': [[0.14, 0.2], [0.34, 0.4]]}
    cns_plot = consistency_plot(contributions_dict=contrib_dict)
    with open("assets/test_consistency_buffered_image.png", "wb") as f:
        f.write(cns_plot.getvalue())
    assert isinstance(cns_plot, io.BytesIO)


if __name__ == '__main__':
    pytest.main(['-sv', __file__])
