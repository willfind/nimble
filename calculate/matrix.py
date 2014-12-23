
from UML.exceptions import ArgumentException

def elementwiseMultiply(left, right):
	"""
	Perform element wise multiplication of two provided UML data objects
	with the result being returned in a separate UML data object. Both
	objects must contain only numeric data. The pointCount and featureCount
	of both objects must be equal. The types of the two objects may be
	different. None is always returned.

	"""
	# check left is UML
	if not isinstance(left, UML.data.Base):
		raise ArgumentException("'left' must be an instance of a UML data object")

	left = left.copy()
	left.elementwiseMultiply(right)
	return left

def elementwisePower(left, right):
	# check left is UML
	if not isinstance(left, UML.data.Base):
		raise ArgumentException("'left' must be an instance of a UML data object")

	left = left.copy()
	left.elementwisePower(right)
	return left
