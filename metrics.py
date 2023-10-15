from sklearn.metrics import accuracy_score


def get_accuracy_score(ground_truth, predictions) -> float:
    if len(ground_truth) != len(predictions):
        return 0
    accuracy = accuracy_score(ground_truth, predictions)
    return accuracy
