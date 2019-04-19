"""
Contains the MultiOutputRidgeRegression custom learner class.
"""

from __future__ import absolute_import

from six.moves import range

import UML
from UML.customLearners import CustomLearner


class MultiOutputRidgeRegression(CustomLearner):
    """
    Learner which will train a version of 'Custom.RidgeRegression' on
    each of the (one or more) features of the prediction data. It then
    matches that output shape when apply() is called.
    """

    learnerType = 'unknown'

    def train(self, trainX, trainY, lamb=0):
        self._learners = []

        for i in range(len(trainY.features)):
            currY = trainY.features.copy(i, useLog=False)

            currTL = UML.train('Custom.RidgeRegression', trainX, currY,
                               lamb=lamb, useLog=False)
            self._learners.append(currTL)

    def apply(self, testX):
        results = None

        for i in range(len(self._learners)):
            curr = self._learners[i].apply(testX, useLog=False)
            if results is None:
                results = curr
            else:
                results.features.add(curr, useLog=False)

        return results
