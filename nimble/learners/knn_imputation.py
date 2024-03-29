
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Contains the KNNImputation learner class.
"""
import numpy as np

import nimble
from nimble import CustomLearner
from nimble.core._learnHelpers import findBestInterface
from nimble.match import anyTrue, _convertMatchToFunction
from nimble.exceptions import InvalidArgumentValue

def _sklPresent():
    try:
        findBestInterface("skl")
    except InvalidArgumentValue:
        return False
    return True

# pylint: disable=attribute-defined-outside-init, arguments-differ
class KNNImputation(CustomLearner):
    """
    Imputation using K-Nearest Neighbors algorithms.
    """
    learnerType = 'transformation'

    def train(self, trainX, trainY=None, k=5, mode=None):
        if mode not in ['classification', 'regression']:
            msg = "mode must be set to 'classification' or 'regression'"
            raise InvalidArgumentValue(msg)
        if _sklPresent():
            self.kwargs = {'n_neighbors': k}
            if mode == 'classification':
                self.learner = 'sciKitLearn.KNeighborsClassifier'
            elif mode == 'regression':
                self.learner = 'sciKitLearn.KNeighborsRegressor'
        else:
            self.kwargs = {'k': k}
            if mode == 'classification':
                self.learner = 'nimble.KNNClassifier'
            elif mode == 'regression':
                # TODO nimble.KNNRegressor
                msg = "nimble does not offer a KNN regressor at this time. "
                msg += "However, installing sci-kit learn will allow for "
                msg += 'regression mode.'
                raise InvalidArgumentValue(msg)

        match = _convertMatchToFunction(np.nan)
        self.match = trainX.matchingElements(match, useLog=False)

    def apply(self, testX):
        testData = testX.copy()
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
                    cannotTrain = map(anyTrue, self.match[:, keepIdx].points)
                    trainIdx = [i for i, v in enumerate(cannotTrain) if not v]
                    toTrain = testData[trainIdx, keepIdx]
                    pred = nimble.trainAndApply(
                        self.learner, toTrain, -1, toPredict, useLog=False,
                        **self.kwargs)
                    newPt[ftIdx] = pred[0]
                    del keepIdx[-1]
            result.append(newPt)

        pnames = testX.points._getNamesNoGeneration()
        fnames = testX.features._getNamesNoGeneration()
        ret = nimble.data(result, pointNames=pnames, featureNames=fnames,
                          returnType=testX.getTypeString(), useLog=False)
        ret._absPath = testX._absPath
        ret._relPath = testX._relPath

        return ret
