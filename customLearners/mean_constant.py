import numpy

import UML
from UML.customLearners import CustomLearner


class MeanConstant(CustomLearner):
    learnerType = 'regression'

    def train(self, trainX, trainY):
        self.mean = trainY.featureStatistics('mean')[0, 0]

    def apply(self, testX):
        raw = numpy.zeros(testX.pointCount)
        numpy.ndarray.fill(raw, self.mean)

        ret = UML.createData("Matrix", raw)
        ret.transpose()
        return ret
