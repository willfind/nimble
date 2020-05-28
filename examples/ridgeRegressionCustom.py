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
    trainX = nimble.createData("Matrix", trainXRaw)
    trainY = nimble.createData("Matrix", trainYRaw)
    testX = nimble.createData("Matrix", testXRaw)
    testY = nimble.createData("Matrix", testYRaw)

    # an example of getting a TrainedLearner and querying its attributes. In
    # RidgeRegression's case, we check the learned coefficients, named 'w'
    trained = nimble.train("nimble.RidgeRegression", trainX, trainY, arguments={'lamb': 0})
    print("Coefficients:")
    print(trained.getAttributes()['w'])

    # Two ways of getting predictions
    pred1 = trained.apply(testX, arguments={'lamb': 0})
    pred2 = nimble.trainAndApply("nimble.RidgeRegression", trainX, trainY, testX, arguments={'lamb': 0})
    assert pred1.isIdentical(pred2)

    # Using cross validation to explicitly determine a winning argument set
    results = nimble.crossValidate("nimble.RidgeRegression", trainX, trainY, RMSE,
                                lamb=nimble.CV([0, .5, 1]))
    bestArguments = results.bestArguments
    bestScore = results.bestResult

    print("Best argument set: " + str(bestArguments))
    print("Best score: " + str(bestScore))

    # Currently, testing can only be done through the top level function trainAndTest()
    # Also: arguments to the learner are given in the python **kwargs style, not as
    # an explicit dict like  seen above.
    # Using lamb = 1 in this case so that there actually are errors
    error = nimble.trainAndTest("nimble.RidgeRegression", trainX, trainY, testX, testY, RMSE, lamb=1)
    print("rootMeanSquareError of predictions with lamb=1: " + str(error))
