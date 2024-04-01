
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
Contains the RidgeRegression custom learner class.
"""

import numpy as np

import nimble
from nimble import CustomLearner
from nimble._utility import dtypeConvert

# pylint: disable=attribute-defined-outside-init, arguments-differ
class RidgeRegression(CustomLearner):
    """
    A least squares regression with a regularization penalty.

    The ``lamb`` parameter determines how much to penalize the size of
    the parameter estimates. When lamb is set to 0, this is the same as
    Ordinary Least Squares (OLS) regression.
    """
    learnerType = 'regression'

    def train(self, trainX, trainY, lamb=0):
        self.lamb = lamb

        # setup for intercept term
        #		ones = nimble.data(np.ones(len(trainX.points)))
        #		ones.transpose()
        #		trainX = trainX.copy()
        #		trainX.features.append(ones)

        # trainX and trainY are input as points in rows, features in columns
        # in other words: Points x Features.
        # for X data, we want both Points x Features and Features x Points
        # for Y data, we only want Points x Features
        rawXPxF = dtypeConvert(trainX.copy(to="numpyarray"))
        rawXFxP = rawXPxF.transpose()
        rawYPxF = dtypeConvert(trainY.copy(to="numpyarray"))

        featureSpace = np.matmul(rawXFxP, rawXPxF)
        lambdaMatrix = lamb * np.identity(len(trainX.features))
        #		lambdaMatrix[len(trainX.features)-1][len(trainX.features)-1] = 0

        inv = np.linalg.inv(featureSpace + lambdaMatrix)
        self.w = np.matmul(np.matmul(inv, rawXFxP), rawYPxF)

    def apply(self, testX):
    # setup intercept
    #		ones = nimble.data(np.ones(len(testX.points)))
    #		ones.transpose()
    #		testX = testX.copy()
    #		testX.features.append(ones)

        # testX input as points in rows, features in columns
        rawXPxF = dtypeConvert(testX.copy(to="numpyarray"))
        rawXFxP = rawXPxF.transpose()

        pred = np.dot(self.w.transpose(), rawXFxP)

        return nimble.data(pred.transpose(), useLog=False)
