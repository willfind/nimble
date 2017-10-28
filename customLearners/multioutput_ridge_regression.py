import UML
from UML.customLearners import CustomLearner


class MultiOutputRidgeRegression(CustomLearner):
    """
    Learner which will train a version of 'Custom.RidgeRegression' on each of
    the (one or more) features of the prediction data. It then matches that
    output shape when apply() is called.

    """

    learnerType = 'unknown'

    def train(self, trainX, trainY, lamb=0):
        self._learners = []

        for i in xrange(trainY.features):
            currY = trainY.copyFeatures(i)

            currTL = UML.train('Custom.RidgeRegression', trainX, currY, lamb=lamb)
            self._learners.append(currTL)

    def apply(self, testX):
        results = None

        for i in xrange(len(self._learners)):
            curr = self._learners[i].apply(testX, useLog=False)
            if results is None:
                results = curr
            else:
                results.appendFeatures(curr)

        return results
