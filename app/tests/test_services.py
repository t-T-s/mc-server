import pytest
from app.services.metrics_service import consistency_scores, consistency_plot, \
    compacity_plot, stability_plot, user_diversity_plot
import io


def test_service_consistency_scores():
    # Basic invocation test
    contrib_dict = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                    'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                    'LIME': [[0.14, 0.2], [0.34, 0.4]]}
    avg, pw_cons = consistency_scores(contributions_dict=contrib_dict)
    print(avg, pw_cons)
    assert isinstance(avg, float)
    assert isinstance(pw_cons, dict)


def test_service_consistency_plot():
    # Basic invocation and saving image file test
    contrib_dict = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                    'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                    'LIME': [[0.14, 0.2], [0.34, 0.4]]}
    cns_plot = consistency_plot(contributions_dict=contrib_dict)
    with open("assets/test_consistency_buffered_image.png", "wb") as f:
        f.write(cns_plot.getvalue())
    assert isinstance(cns_plot, io.BytesIO)


def test_service_compacity_plot():
    # Basic invocation test
    contributions = [[0.15, 0.2, 0.4, 0.01], [0.3, 0.42, 0.34, 0.012], [0.3, 0.42, 0.24, 0.011]]
    selection = [0, 1, 2]
    distance = 0.9
    nb_features = 2
    comp_plot = compacity_plot(contributions, selection, distance, nb_features)
    with open("assets/test_compacity_buffered_image.png", "wb") as f:
        f.write(comp_plot.getvalue())
    assert isinstance(comp_plot, io.BytesIO)


# def test_service_stability_plot():
#     # Basic invocation test
#     selection = None
#     max_points = 500
#     max_features = 2
#     # The number of inputs have to be greater than 10
#     x = [[0.46582937, 0.36313128, 0.17189367, 0.01546506],
#          [0.507121, 0.01918931, 0.78464877, 0.77427306],
#          [0.50496966, 0.98061098, 0.27967449, 0.29417485],
#          [0.3521993, 0.48835738, 0.23460793, 0.64657831],
#          [0.67447758, 0.39336409, 0.01272322, 0.06723874],
#          [0.64747419, 0.46749566, 0.35986405, 0.95362188],
#          [0.68757052, 0.66098619, 0.99523119, 0.09020147],
#          [0.50099671, 0.93649314, 0.25915279, 0.75566948],
#          [0.87476536, 0.04664153, 0.89152254, 0.49654976],
#          [0.87059629, 0.86237521, 0.85991116, 0.08158515],
#          [0.70103771, 0.28800044, 0.20746705, 0.44251794]]
#     contributions = [[0.40302247, 0.2380319, 0.2301524, 0.51886267],
#                      [0.67702666, 0.56105417, 0.2590706, 0.16573209],
#                      [0.89889708, 0.20274018, 0.35884345, 0.81657564],
#                      [0.90689996, 0.47364288, 0.7311526, 0.62089024],
#                      [0.40710054, 0.14887248, 0.84189794, 0.71489193],
#                      [0.08248856, 0.35354858, 0.11228026, 0.99185406],
#                      [0.16592895, 0.53207895, 0.33886526, 0.5177407],
#                      [0.81840455, 0.52111217, 0.81772124, 0.15083058],
#                      [0.84590188, 0.52619182, 0.89583799, 0.21780331],
#                      [0.09295941, 0.83894879, 0.46052668, 0.1308371],
#                      [0.07235874, 0.92571017, 0.05129698, 0.92341386]]
#     y_target = [[1], [1], [0], [1], [0], [0], [0], [1], [0], [1], [1]]
#     stabl_plot = stability_plot(
#         X=x
#         , contributions=contributions
#         , y_target=y_target
#         , selection=selection
#         , max_points=max_points
#         , max_features=max_features)
#     with open("assets/test_stability_buffered_image.png", "wb") as f:
#         f.write(stabl_plot.getvalue())
#     assert isinstance(stabl_plot, io.BytesIO)


def test_service_user_diversity_plot():
    # Basic invocation test
    predictions = [[-2.218350887298584, -2.198277711868286],
                   [-2.5687193870544434, -2.458390474319458],
                   [-2.0745654106140137, -2.329625368118286],
                   [-2.2383768558502197, -2.222764253616333],
                   [-2.269338846206665, -2.456698179244995],
                   [-1.9543006420135498, -2.549536943435669]]
    client_ids = [1, 1, 1, 2, 2, 2]
    perplexity = 4
    diversity_plt = user_diversity_plot(predictions, client_ids, perplexity)
    with open("assets/test_user_diversity_image.png", "wb") as f:
        f.write(diversity_plt.getvalue())
    assert isinstance(diversity_plt, io.BytesIO)


if __name__ == '__main__':
    pytest.main(['-sv', __file__])
