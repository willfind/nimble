import numpy

import UML
from UML.customLearners import CustomLearner


class RidgeRegression(CustomLearner):
    learnerType = 'regression'

    def train(self, trainX, trainY, lamb=0):
        self.lamb = lamb

        # setup for intercept term
        #		ones = UML.createData("Matrix", numpy.ones(trainX.pointCount))
        #		ones.transpose()
        #		trainX = trainX.copy()
        #		trainX.appendFeatures(ones)

        # trainX and trainY are input as points in rows, features in columns
        # in other words: Points x Features.
        # for X data, we want both Points x Features and Features x Points
        # for Y data, we only want Points x Features
        rawXPxF = trainX.copyAs("numpymatrix")
        rawXFxP = rawXPxF.transpose()
        rawYPxF = trainY.copyAs("numpymatrix")

        featureSpace = rawXFxP * rawXPxF
        lambdaMatrix = lamb * numpy.identity(trainX.featureCount)
        #		lambdaMatrix[trainX.featureCount-1][trainX.featureCount-1] = 0

        inv = numpy.linalg.inv(featureSpace + lambdaMatrix)
        self.w = inv * rawXFxP * rawYPxF

    def apply(self, testX):
    # setup intercept
    #		ones = UML.createData("Matrix", numpy.ones(testX.pointCount))
    #		ones.transpose()
    #		testX = testX.copy()
    #		testX.appendFeatures(ones)

        # testX input as points in rows, features in columns
        rawXPxF = testX.copyAs("numpyarray")
        rawXFxP = rawXPxF.transpose()

        pred = numpy.dot(self.w.transpose(), rawXFxP)

        return UML.createData("Matrix", pred.transpose())
