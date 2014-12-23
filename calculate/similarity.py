
import UML
import numpy

from UML.exceptions import ArgumentException

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
