
import UML
import numpy

from UML.exceptions import ArgumentException
from UML.calculate import fractionIncorrect

def _validatePredictedAsLabels(predictedValues):
	if not isinstance(predictedValues, UML.data.Base):
		raise ArgumentException("predictedValues must be derived class of UML.data.Base")
	if predictedValues.featureCount > 1:
		raise ArgumentException("predictedValues must be labels only; this has more than one feature")

def cosineSimilarity(knownValues, predictedValues):
	_validatePredictedAsLabels(predictedValues)

	known = knownValues.copyAs(format="numpy array").flatten()
	predicted = predictedValues.copyAs(format="numpy array").flatten()

	numerator = (numpy.dot(known, predicted))
	denominator = (numpy.linalg.norm(known) * numpy.linalg.norm(predicted))

	return numerator / denominator


def correlation(X, X_T=None):
	if X_T is None:
		X_T = X.copy()
		X_T.transpose()
	stdVector = X.pointStatistics('populationstd')
	stdVector_T = stdVector.copy()
	stdVector_T.transpose()

	cov = covariance(X, X_T, False)
	stdMatrix = stdVector * stdVector_T
	ret = cov / stdMatrix

	return ret

def covariance(X, X_T=None, sample=True):
	if X_T is None:
		X_T = X.copy()
		X_T.transpose()
	pointMeansVector = X.pointStatistics('mean')
	fill = lambda x: [x[0]] * X.featureCount
	pointMeans = pointMeansVector.applyToPoints(fill, inPlace=False)
	pointMeans_T = pointMeans.copy()
	pointMeans_T.transpose()

	XminusEofX = X - pointMeans
	X_TminusEofX_T = X_T - pointMeans_T

	# doing sample covariance calculation
	if sample:
		divisor = X.featureCount - 1
	# doing population covariance calculation
	else:
		divisor = X.featureCount

	ret = (XminusEofX * X_TminusEofX_T) / divisor
	return ret

def fractionCorrect(knownValues, predictedValues):
	return 1 - fractionIncorrect(knownValues, predictedValues)
