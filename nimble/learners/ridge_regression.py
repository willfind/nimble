"""
Contains the RidgeRegression custom learner class.
"""

import numpy

import nimble
from nimble import CustomLearner
from nimble.utility import dtypeConvert


class RidgeRegression(CustomLearner):
    learnerType = 'regression'

    def train(self, trainX, trainY, lamb=0):
        self.lamb = lamb

        # setup for intercept term
        #		ones = nimble.data("Matrix", numpy.ones(len(trainX.points)))
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

        featureSpace = numpy.matmul(rawXFxP, rawXPxF)
        lambdaMatrix = lamb * numpy.identity(len(trainX.features))
        #		lambdaMatrix[len(trainX.features)-1][len(trainX.features)-1] = 0

        inv = numpy.linalg.inv(featureSpace + lambdaMatrix)
        self.w = numpy.matmul(numpy.matmul(inv, rawXFxP), rawYPxF)

    def apply(self, testX):
    # setup intercept
    #		ones = nimble.data("Matrix", numpy.ones(len(testX.points)))
    #		ones.transpose()
    #		testX = testX.copy()
    #		testX.features.append(ones)

        # testX input as points in rows, features in columns
        rawXPxF = dtypeConvert(testX.copy(to="numpyarray"))
        rawXFxP = rawXPxF.transpose()

        pred = numpy.dot(self.w.transpose(), rawXFxP)

        return nimble.data("Matrix", pred.transpose(), useLog=False)
