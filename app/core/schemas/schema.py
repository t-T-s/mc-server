from pydantic import BaseModel, model_validator
from typing import List, Dict, Union
from app.utils.mc_exceptions import ShapeError


# pydantic models

class ClfLabels(BaseModel):
    ground_truth: Union[List[int], List[List[int]]] = [1, 2, 3, 3, 5]
    predictions: Union[List[int], List[List[int]]] = [1, 2, 3, 4, 5]

    @model_validator(mode='after')
    def check_label_indicator_columns(self) -> 'ClfLabels':
        # get all fields in the model
        fields = self.model_dump()  # returns a dict
        assert len(self.ground_truth) == len(self.predictions), 'Input arrays are of different lengths'

        # check if the lists are nested and the inner lists are of same length
        if any(isinstance(item, list) for item in self.ground_truth) and \
                any(isinstance(item, list) for item in self.predictions):
            for _, value in fields.items():
                if not all(len(sub) == 2 for sub in value):
                    raise ShapeError('For a multi class labels the label indicator array should have the same '
                                     'lengths for all the items')
        return self


class ContributionsDict(BaseModel):
    # contribution_dict = {'KernelSHAP': contrib_KernelSHAP, 'SamplingSHAP': contrib_SamplingSHAP
    #  'LIME': contrib_LIME }
    contribution_dict: Dict[str, List[List[float]]] = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                                                       'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                                                       'LIME': [[0.14, 0.2], [0.34, 0.4]]}

    @model_validator(mode='after')
    def check_feature_columns(self) -> 'ContributionsDict':
        # get all fields in the model
        # check if the internal lists are of same length
        for _, value in self.contribution_dict.items():
            if all(len(sub) == value[0] for sub in value):
                raise ShapeError('Contribution rows must be of same length for all the rows')
        return self


class Contributions(BaseModel):
    # Contributions from one type of explainer
    contributions: List[List[float]] = [[0.15, 0.2, 0.4, 0.01], [0.3, 0.42, 0.34, 0.012]]
    selection: Union[List[int], None] = [0, 1]
    distance: Union[float, None] = 0.9
    nb_features: Union[int, None] = 2

    @model_validator(mode='after')
    def check_feature_columns(self) -> 'Contributions':
        # get all fields in the model
        # check if the internal lists are of same length
        if all(len(sub) == self.contributions[0] for sub in self.contributions):
            raise ShapeError('Contribution rows must be of same length for all the rows')
        return self


class StabilityData(BaseModel):
    # Contributions from one type of explainer
    x_input: List[List[float]] = [[0.46582937, 0.36313128, 0.17189367, 0.01546506],
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
    contributions: List[List[float]] = [[0.40302247, 0.2380319, 0.2301524, 0.51886267],
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
    y_target: List[List[float]] = [[1], [1], [0], [1], [0], [0], [0], [1], [0], [1], [1]]
    selection: Union[List[int], None] = [0, 1]
    max_points: Union[int, None] = 500
    max_features: Union[int, None] = 2

    @model_validator(mode='after')
    def check_feature_columns(self) -> 'StabilityData':
        # get all fields in the model
        # check if the internal lists are of same length
        if len(self.contributions) < 10 or len(self.x_input) < 10 or len(self.y_target) < 10:
            raise NotImplementedError("The stability plot is only implemented for X data size of 10 rows or more. " +
                                      "The contributions and y_target should be also of the same length")
        if all(len(sub) == self.contributions[0] for sub in self.contributions):
            raise ShapeError('Contribution rows must be of same length for all the rows')
        if all(len(sub) == self.x_input[0] for sub in self.x_input):
            raise ShapeError('Input X data rows must be of same length for all the rows')
        if all(len(sub) == self.y_target[0] for sub in self.y_target):
            raise ShapeError('Input y target rows must be of same length for all the rows')
        return self


class MetricResponse(BaseModel):
    metric_value: dict


class UserDiversityInput(BaseModel):
    predictions: List[List[float]] = [[-2.218350887298584, -2.198277711868286],
                                      [-2.5687193870544434, -2.458390474319458],
                                      [-2.0745654106140137, -2.329625368118286],
                                      [-2.2383768558502197, -2.222764253616333],
                                      [-2.269338846206665, -2.456698179244995],
                                      [-1.9543006420135498, -2.549536943435669]]

    client_ids: List[int] = [1, 1, 1, 2, 2, 2]
    perplexity: int

    @model_validator(mode='after')
    def check_user_diversity(self) -> 'UserDiversityInput':
        # get all fields in the model
        # check if the internal lists are of same length
        if len(self.predictions) != len(self.client_ids):
            raise ShapeError('Predictions rows must be of same length for all the client IDs')
        if self.perplexity >= len(self.predictions):
            raise ValueError('Perplexity must be less than total number of client IDs')
        return self
