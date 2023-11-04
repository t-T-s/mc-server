from pydantic import BaseModel, model_validator
from typing import List, Dict, Union


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
                output = all(len(sub) == 2 for sub in value)
                assert output, ('For a multi class labels the label indicator array should have the same lengths for '
                                'all the items')
        return self


class Contributions(BaseModel):
    # contribution_dict = {'KernelSHAP': contrib_KernelSHAP, 'SamplingSHAP': contrib_SamplingSHAP
    #  'LIME': contrib_LIME }
    contribution_dict: Dict[str, List[List[float]]] = {'KernelSHAP': [[0.15, 0.2], [0.3, 0.42]],
                                                       'SamplingSHAP': [[0.12, 0.2], [0.32, 0.4]],
                                                       'LIME': [[0.14, 0.2], [0.34, 0.4]]}

    @model_validator(mode='after')
    def check_feature_columns(self) -> 'Contributions':
        # get all fields in the model
        # check if the internal lists are of same length
        for _, value in self.contribution_dict.items():
            output = all(len(sub) == 2 for sub in value)
            assert output, ('For a multi class labels the label indicator array should have the same lengths for all '
                            'the items')
        return self


class MetricResponse(BaseModel):
    metric_value: dict
