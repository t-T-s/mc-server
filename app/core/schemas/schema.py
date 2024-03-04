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
        if len(self.perplexity) >= len(self.predictions):
            raise ValueError('Perplexity must be less than total number of client IDs')
        return self