class Learner:
    def __init__(self):
        self.reset_time = 3600
        self.confidence_in_interval = []

    def find_similar_prediction_shapes(self, current_predictions, new_prediction, iou):
        return [prediction
                for prediction in current_predictions
                if prediction.category_id == new_prediction.category_id
                and prediction.intersection_over_union(new_prediction) >= iou
                ]
