"""
Script demonstrating the training, applying, and testing api using an
out-of-the-box, custom learner.

"""

from __future__ import absolute_import
from __future__ import print_function
try:
    from allowImports import boilerplate
except:
    from .allowImports import boilerplate

boilerplate()

if __name__ == "__main__":
    import UML
    from UML.calculate import rootMeanSquareError as RMSE
    from UML.randomness import numpyRandom

    # produce some simple linear data
    trainPoints = 10
    testPoints = 5
    feats = 3
    targetCoefs = numpyRandom.rand(feats, 1)
    trainXRaw = numpyRandom.randint(-10, 10, (trainPoints, feats))
    trainYRaw = trainXRaw.dot(targetCoefs)
    testXRaw = numpyRandom.randint(-10, 10, (testPoints, feats))
    testYRaw = testXRaw.dot(targetCoefs)

    # encapsulate in UML data objects
    trainX = UML.createData("Matrix", trainXRaw, name="trainingData")
    trainY = UML.createData("Matrix", trainYRaw, name="trainingLabels")
    testX = UML.createData("Matrix", testXRaw, name="testingData")
    testY = UML.createData("Matrix", testYRaw, name="testingLabels")

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

    # UML.log("randomString", "I am logging a                                    reasonably long                        string to make sure that everything in the code that I wrote is working correctly so lets see how this goes ok bye")
    # UML.log("randomList",[1, "g", 9, 888, "hello", "bye"]*5)
    # UML.log("randomDict", {"this": "is", "a":1, "random":"dictionary", "with":"many", 5:10000000000, 6:334040404040, "different":"keys"})
    # trainX[:,0].featureReport()
    # trainX.summaryReport()
    UML.showLog(levelOfDetail=3, mostRunsAgo=6, saveToFileName="test.txt")
