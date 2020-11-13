"""
Contains the MeanConstant custom learner class.
"""

import numpy

import nimble
from nimble import CustomLearner

class MeanConstant(CustomLearner):
    learnerType = 'regression'

    def train(self, trainX, trainY):
        self.mean = trainY.features.statistics('mean')[0, 0]

    def apply(self, testX):
        raw = numpy.zeros(len(testX.points))
        numpy.ndarray.fill(raw, self.mean)

        ret = nimble.data("Matrix", raw, useLog=False)
        ret.transpose(useLog=False)
        return ret
