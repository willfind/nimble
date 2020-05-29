"""
Contains the MultiOutputLinearRegression custom learner class.
"""

import numpy
try:
    from sklearn.linear_model import LinearRegression
    imported = True
except ImportError:
    imported = False

import nimble
from nimble import CustomLearner
from nimble.utility import numpy2DArray, dtypeConvert


class MultiOutputLinearRegression(CustomLearner):
    """
    Learner which trains a separate linear regerssion model on each of
    the features of the prediction data. The backend learner is provided
    by scikit-learn.
    """

    learnerType = 'regression'

    def train(self, trainX, trainY):
        self._models = []
        rawTrainX = dtypeConvert(trainX.copy(to='numpyarray'))

        for i in range(len(trainY.features)):
            currY = trainY.features.copy(i, useLog=False)
            rawCurrY = dtypeConvert(currY.copy(to='numpyarray',
                                    outputAs1D=True))

            currModel = LinearRegression()
            currModel.fit(rawTrainX, rawCurrY)
            self._models.append(currModel)

    def apply(self, testX):
        results = []
        rawTestX = dtypeConvert(testX.copy(to='numpyarray'))

        for i in range(len(self._models)):
            curr = self._models[i].predict(rawTestX)
            results.append(curr)

        results = numpy2DArray(results)
        results = results.transpose()

        return nimble.data("Matrix", results, useLog=False)
