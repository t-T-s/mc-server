from pydantic import BaseModel, model_validator
from typing import List, Union


# pydantic models

class ClfLabels(BaseModel):
    ground_truth: Union[List[int], List[List[int]]] = [1, 2, 3, 4, 5]
    predictions: Union[List[int], List[List[int]]] = [1, 2, 3, 4, 5]

    @model_validator(mode='after')
    def check_label_indicator_column_consistency(self) -> 'ClfLabels':
        # get all fields in the model
        fields = self.model_dump()  # returns a dict
        assert len(self.ground_truth) == len(self.predictions), 'Input lengths are different'

        # check if the internal lists are of same length
        for field, value in fields.items():
            output = all(len(sub) == 2 for sub in value)
            assert output, 'For a multi class labels the label indicator array should have the same lengths for all the nested lists'
        return self


class MetricResponse(BaseModel):
    metric_value: dict
