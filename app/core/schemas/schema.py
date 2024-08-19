from pydantic import BaseModel, model_validator
from typing import List, Dict, Union, Any
from fastapi import HTTPException
import _io


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
                    raise HTTPException(status_code=422, detail='For a multi class labels the label indicator array '
                                                                'should have the same'
                                                                'lengths for all the items')
        return self


class ContributionsDict(BaseModel):
    contribution_dict: Dict[str, List[List[float]]] = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                                                       'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                                                       'LIME': [[0.14, 0.2], [0.34, 0.4]]}

    @model_validator(mode='after')
    def check_feature_columns(self) -> 'ContributionsDict':
        # get all fields in the model
        # check if the internal lists are of same length
        # Initialize variables to store the expected number of rows and row length
        first_key = list(self.contribution_dict.keys())[0]
        expected_row_count = len(self.contribution_dict[first_key])
        expected_row_length = len(self.contribution_dict[first_key][0])

        # Check if all values have the same shape
        same_shape = True

        for key, value in self.contribution_dict.items():
            if len(value) != expected_row_count:
                same_shape = False
                break
            if any(len(row) != expected_row_length for row in value):
                same_shape = False
                break
        if not same_shape:
            raise HTTPException(status_code=422, detail='Contribution rows must be of same length for all the rows')
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
        if len(set(len(row) for row in self.contributions)) > 1:
            raise HTTPException(status_code=422, detail='Contribution rows must be of same length for all the rows')
        return self


class StabilityData(BaseModel):
    x_train: Union[List[List[float]], None] = [[0.24763825, 0.4624466, 0.1439733, 0.63356432],
                                  [0.89960405, 0.60607923, 0.58054955, 0.07378852],
                                  [0.91335706, 0.77419346, 0.70098694, 0.08870475],
                                  [0.46562798, 0.85730786, 0.53299792, 0.84305255],
                                  [0.32298965, 0.86707571, 0.73935329, 0.8347728]]
    y_train: Union[List[List[float]], None] = [[1], [1], [1], [1], [0]]
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
    pre_trained_model: Union[None, bytes] = None

    @model_validator(mode='after')
    def check_feature_columns(self) -> 'StabilityData':
        # get all fields in the model
        # check if the internal lists are of same length
        if len(self.contributions) < 10 or len(self.x_input) < 10 or len(self.y_target) < 10:
            raise HTTPException(status_code=422, detail="The stability plot is only implemented for contributions "
                                                        "size of 10 rows or more. Thereby x_test and y_target must be "
                                                        "of the same size")
        if not (len(self.contributions) == len(self.x_input) == len(self.y_target)):
            raise HTTPException(status_code=422, detail="The contributions, x_input and y_target should be of "
                                                        "the same length")
        if len(self.x_train) != len(self.y_train):
            raise HTTPException(status_code=422, detail="The training dataset x_train and y_train should be of same "
                                                        "length")
        if len(set(len(row) for row in self.contributions)) > 1:
            raise HTTPException(status_code=422, detail='Contribution rows must be of same length for all the rows')
        if len(set(len(row) for row in self.x_input)) > 1:
            raise HTTPException(status_code=422, detail='Input X data rows must be of same length for all the rows')
        if len(set(len(row) for row in self.y_target)) > 1:
            raise HTTPException(status_code=422, detail='Input y target rows must be of same length for all the rows')
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
            raise HTTPException(status_code=422,
                                detail='Predictions rows must be of same length for all the client IDs')
        if self.perplexity >= len(self.predictions):
            raise HTTPException(status_code=422, detail='Perplexity must be less than total number of client IDs')
        return self
