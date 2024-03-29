
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
Script demonstrating the training, applying, and testing api using an
out-of-the-box, custom learner.
"""

import nimble
from nimble.calculate import rootMeanSquareError as RMSE
from nimble.random import numpyRandom

if __name__ == "__main__":
    # produce some simple linear data
    trainPoints = 10
    testPoints = 5
    feats = 3
    targetCoefs = numpyRandom.rand(feats, 1)
    trainXRaw = numpyRandom.randint(-10, 10, (trainPoints, feats))
    trainYRaw = trainXRaw.dot(targetCoefs)
    testXRaw = numpyRandom.randint(-10, 10, (testPoints, feats))
    testYRaw = testXRaw.dot(targetCoefs)

    # encapsulate in nimble Base objects
    trainX = nimble.data(trainXRaw)
    trainY = nimble.data(trainYRaw)
    testX = nimble.data(testXRaw)
    testY = nimble.data(testYRaw)

    # an example of getting a TrainedLearner and querying its attributes. In
    # RidgeRegression's case, we check the learned coefficients, named 'w'
    trained = nimble.train("nimble.RidgeRegression", trainX, trainY, arguments={'lamb': 0})
    print("Coefficients:")
    print(trained.getAttributes()['w'])

    # Two ways of getting predictions
    pred1 = trained.apply(testX)
    pred2 = nimble.trainAndApply("nimble.RidgeRegression", trainX, trainY, testX, arguments={'lamb': 0})
    assert pred1.isIdentical(pred2)

    # Using cross validation to explicitly determine a winning argument set
    results = nimble.train("nimble.RidgeRegression", trainX, trainY,
                           tuning=RMSE, lamb=nimble.Tune([0, .5, 1]))
    bestArguments = results.tuning.bestArguments
    bestScore = results.tuning.bestResult

    print("Best argument set: " + str(bestArguments))
    print("Best score: " + str(bestScore))

    # Currently, testing can only be done through the top level function trainAndTest()
    # Also: arguments to the learner are given in the python **kwargs style, not as
    # an explicit dict like  seen above.
    # Using lamb = 1 in this case so that there actually are errors
    error = nimble.trainAndTest("nimble.RidgeRegression", RMSE, trainX, trainY, testX, testY, lamb=1)
    print("rootMeanSquareError of predictions with lamb=1: " + str(error))
