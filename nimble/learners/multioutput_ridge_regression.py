
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
Contains the MultiOutputRidgeRegression custom learner class.
"""

import nimble
from nimble import CustomLearner

# pylint: disable=attribute-defined-outside-init, arguments-differ
class MultiOutputRidgeRegression(CustomLearner):
    """
    Learner which will train a version of 'nimble.RidgeRegression' on
    each of the (one or more) features of the prediction data. It then
    matches that output shape when apply() is called.
    """

    learnerType = 'regression'

    def train(self, trainX, trainY, lamb=0):
        self._learners = []

        for i in range(len(trainY.features)):
            currY = trainY.features.copy(i, useLog=False)

            currTL = nimble.train('nimble.RidgeRegression', trainX, currY,
                                  lamb=lamb, useLog=False)
            self._learners.append(currTL)

    def apply(self, testX):
        results = None

        for learner in self._learners:
            curr = learner.apply(testX, useLog=False)
            if results is None:
                results = curr
            else:
                results.features.append(curr, useLog=False)

        return results
