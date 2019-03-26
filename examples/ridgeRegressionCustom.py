"""
Script demonstrating the training, applying, and testing api using an
out-of-the-box, custom learner.

"""

import UML
from UML.calculate import rootMeanSquareError as RMSE
from UML.randomness import numpyRandom

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

    # encapsulate in UML Base objects
    trainX = UML.createData("Matrix", trainXRaw)
    trainY = UML.createData("Matrix", trainYRaw)
    testX = UML.createData("Matrix", testXRaw)
    testY = UML.createData("Matrix", testYRaw)

    # an example of getting a TrainedLearner and querying its attributes. In
    # RidgeRegression's case, we check the learned coefficients, named 'w'
    trained = UML.train("custom.RidgeRegression", trainX, trainY, arguments={'lamb': 0})
    print("Coefficients:")
    print(trained.getAttributes()['w'])

    # Two ways of getting predictions
    pred1 = trained.apply(testX, arguments={'lamb': 0})
    pred2 = UML.trainAndApply("custom.RidgeRegression", trainX, trainY, testX, arguments={'lamb': 0})
    assert pred1.isIdentical(pred2)

    # Using cross validation to explicitly determine a winning argument set
    results = UML.crossValidateReturnBest("custom.RidgeRegression", trainX, trainY, RMSE,
                                          lamb=(0, .5, 1))
    bestArgument, bestScore = results

    print("Best argument set: " + str(bestArgument))
    print("Best score: " + str(bestScore))

    # Currently, testing can only be done through the top level function trainAndTest()
    # Also: arguments to the learner are given in the python **kwargs style, not as
    # an explicit dict like  seen above.
    # Using lamb = 1 in this case so that there actually are errors
    error = UML.trainAndTest("custom.RidgeRegression", trainX, trainY, testX, testY, RMSE, lamb=1)
    print("rootMeanSquareError of predictions with lamb=1: " + str(error))
