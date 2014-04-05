
import numpy

import UML
from UML.interfaces.custom_learner import CustomLearner

class RidgeRegression(CustomLearner):

	learnerType = 'Regression'

	def train(self, trainX, trainY, lamb=0):
		self.lamb = lamb

#		ones = UML.createData("Matrix", numpy.ones(trainX.pointCount))
#		ones.transpose()
#		trainX.appendFeatures(ones)

		# trainX and trainY are input as points in rows, features in columns
		rawXPxF = trainX.copyAs("numpymatrix")
		rawXFxP = rawXPxF.transpose()
		rawYPxF = trainY.copyAs("numpymatrix")
		
		featureSpace = rawXFxP * rawXPxF
		lambdaMatrix = lamb * numpy.identity(trainX.featureCount)
#		lambdaMatrix[trainX.featureCount-1][trainX.featureCount-1] = 0

		inv = numpy.linalg.inv(featureSpace + lambdaMatrix)
		self.w = inv * rawXFxP * rawYPxF

	def apply(self, testX):
#		ones = UML.createData("Matrix", numpy.ones(testX.pointCount))
#		ones.transpose()
#		testX.appendFeatures(ones)

		# testX input as points in rows, features in columns
		rawXPxF = testX.copyAs("numpyarray")
		rawXFxP = rawXPxF.transpose()

		pred = numpy.dot(self.w.transpose(), rawXFxP)

		return UML.createData("Matrix", pred.transpose())
