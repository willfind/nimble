
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
Contains the MultiOutputLinearRegression custom learner class.
"""

try:
    from sklearn.linear_model import LinearRegression
    imported = True
except ImportError:
    imported = False

import nimble
from nimble import CustomLearner
from nimble._utility import numpy2DArray, dtypeConvert

# pylint: disable=attribute-defined-outside-init
class MultiOutputLinearRegression(CustomLearner):
    """
    Learner which trains a separate linear regression model on each of
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

        for model in self._models:
            curr = model.predict(rawTestX)
            results.append(curr)

        results = numpy2DArray(results)
        results = results.transpose()

        return nimble.data(results, useLog=False)
