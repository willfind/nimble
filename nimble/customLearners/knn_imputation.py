
import numpy

import nimble
from nimble.helpers import findBestInterface
from nimble.match import anyTrue, convertMatchToFunction
from nimble.exceptions import InvalidArgumentValue

from .custom_learner import CustomLearner

def sklPresent():
    try:
        findBestInterface("skl")
    except InvalidArgumentValue:
        return False
    return True

class KNNImputation(CustomLearner):
    learnerType = 'unknown'

    def train(self, trainX, trainY=None, k=5, mode=None):
        if mode not in ['classification', 'regression']:
            msg = "mode must be set to 'classification' or 'regression'"
            raise InvalidArgumentValue(msg)
        if sklPresent():
            self.kwargs = {'n_neighbors': k}
            if mode == 'classification':
                self.learner = 'sciKitLearn.KNeighborsClassifier'
            elif mode == 'regression':
                self.learner = 'sciKitLearn.KNeighborsRegressor'
        else:
            self.kwargs = {'k': k}
            if mode == 'classification':
                self.learner = 'Custom.KNNClassifier'
            elif mode == 'regression':
                # TODO Custom.KNNRegressor
                msg = "nimble does not offer a KNN regressor at this time. "
                msg += "However, installing sci-kit learn will allow for "
                msg += 'regression mode.'
                raise InvalidArgumentValue(msg)

        match = convertMatchToFunction(numpy.nan)
        self.match = trainX.matchingElements(match, useLog=False)

    def apply(self, testX):
        testData = testX.copy()
        testNames = testData.points.getNames()
        self.match.points.setNames(testNames, useLog=False)
        result = []
        for pt, fill in zip(testData.points, self.match.points):
            newPt = list(pt)
            if anyTrue(fill):
                fillIdx = []
                keepIdx = []
                for idx, val in enumerate(fill):
                    if val:
                        fillIdx.append(idx)
                    else:
                        keepIdx.append(idx)
                toPredict = pt[keepIdx]
                for ftIdx in fillIdx:
                    keepIdx.append(ftIdx)
                    canTrain = self.match[:, keepIdx]
                    canTrain.points.delete(anyTrue, useLog=False)
                    toTrain = testData[canTrain.points.getNames(), keepIdx]
                    pred = nimble.trainAndApply(
                        self.learner, toTrain, -1, toPredict, useLog=False,
                        **self.kwargs)
                    newPt[ftIdx] = pred[0]
                    del keepIdx[-1]
            result.append(newPt)

        pnames = testX.points._getNamesNoGeneration()
        fnames = testX.features._getNamesNoGeneration()
        paths = (testX._absPath, testX._relPath)
        return nimble.createData(testX.getTypeString(), result,
                                 pointNames=pnames, featureNames=fnames,
                                 path=paths, useLog=False)
