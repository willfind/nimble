"""
Contains the MeanConstant custom learner class.
"""

from __future__ import absolute_import

import numpy

import UML
from UML.customLearners import CustomLearner


class MeanConstant(CustomLearner):
    learnerType = 'regression'

    def train(self, trainX, trainY):
        self.mean = trainY.features.statistics('mean')[0, 0]

    def apply(self, testX):
        raw = numpy.zeros(len(testX.points))
        numpy.ndarray.fill(raw, self.mean)

        ret = UML.createData("Matrix", raw, useLog=False)
        ret.transpose(useLog=False)
        return ret
