
from predictions import Prediction, PredictionImpossible


class AlgsBase:
    """Abstract class where is defined the basic behavior of a prediction
    algorithm.
    """
    def __init__(self, **kwargs):
        pass

    def fit(self, trainset):
        """Train an algorithm on a given training set.
        """

        self.trainset = trainset
        return self

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        """Compute the rating prediction for given user and item.
        """

        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = "UKN__" + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = "UKN__" + str(iid)

        details = {}
        try:
            est = self.estimate(iuid, iiid)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details["was_impossible"] = False

        except PredictionImpossible as e:
            est = self.default_prediction()
            details["was_impossible"] = True
            details["reason"] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def default_prediction(self):
        """Used when the ``PredictionImpossible`` exception is raised during a
        call to predict() method
        """
        # The mean of all ratings in the trainset
        return self.trainset.global_mean

    def test(self, testset, verbose=False):
        """Test the algorithm on given testset, i.e. estimate all the ratings
        in the given testset.
        """

        # The ratings are translated back to their original scale.
        predictions = [
            self.predict(uid, iid, r_ui_trans, verbose=verbose)
            for (uid, iid, r_ui_trans) in testset
        ]
        return predictions
