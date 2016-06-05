"""
Defines a subclass of the base data object, which serves as the primary
base class for read only views of data objects.

"""

from base import Base
from UML.exceptions import ImproperActionException

class BaseView(Base):
	"""
	Class defining read only view objects, which have the same api as a
	normal data object, but disallow all methods which could change the
	data.

	"""

	def __init__(self, source, pointStart, pointEnd, featureStart, featureEnd,
				**kwds):
		"""
		Initializes the object which overides all of the funcitonality in
		UML.data.Base to either handle the provided access limits or throw
		exceptions for inappopriate operations.

		source: the UML object that this is a view into.

		pointStart: the inclusive index of the first point this view will have
		access to.

		pointEnd: the EXCLUSIVE index defining the last point this view will
		have access to. This internal representation cannot match the style
		of the factory method (in which both start and end are inclusive)
		because we must be able to define empty ranges by having start = end

		featureStart: the inclusive index of the first feature this view will
		have access to.

		featureEnd: the EXCLUSIVE index defining the last feature this view
		will have access to. This internal representation cannot match the
		style of the factory method (in which both start and end are inclusive)
		because we must be able to define empty ranges by having start = end

		kwds: included due to best practices so args may automatically be
		passed further up into the hierarchy if needed.

		"""
		self._source = source
		self._pStart = pointStart
		self._pEnd = pointEnd
		self._fStart = featureStart
		self._fEnd = featureEnd
		kwds['name'] = self._source.name
		super(BaseView, self).__init__(**kwds)

	# redifinition from Base, except without the setter, using source
	# object's attributes
#	def _getObjName(self):
#		return self._source._name
#	name = property(_getObjName, doc="A name to be displayed when printing or logging this object")

	# redifinition from Base, using source object's attributes
	def _getAbsPath(self):
		return self._source._absPath
	absolutePath = property(_getAbsPath, doc="The path to the file this data originated from, in absolute form")

	# redifinition from Base, using source object's attributes
	def _getRelPath(self):
		return self._source._relPath
	relativePath = property(_getRelPath, doc="The path to the file this data originated from, in relative form")


# TODO: retType


	############################
	# Reimplemented Operations #
	############################

	def getPointNames(self):
		"""Returns a list containing all point names, where their index
		in the list is the same as the index of the point they correspond
		to.

		"""
		ret = self._source.getPointNames()
		ret = ret[self._pStart:self._pEnd]

		return ret

	def getFeatureNames(self):
		"""Returns a list containing all feature names, where their index
		in the list is the same as the index of the feature they
		correspond to.

		"""
		ret = self._source.getFeatureNames()
		ret = ret[self._fStart:self._fEnd]
		
		return ret

	def getPointName(self, index):
		corrected = index + self._pStart
		return self._source.pointNamesInverse[index]

	def getPointIndex(self, name):
		possible = self._source.getPointIndex(name)
		if possible >= self._pStart and possible < self._pEnd:
			return possible
		else:
			raise KeyError()

	def getFeatureName(self, index):
		corrected = index + self._fStart
		return self._source.featureNamesInverse[index]

	def getFeatureIndex(self, name):
		possible = self._source.getFeatureIndex(name)
		if possible >= self._fStart and possible < self._fEnd:
			return possible
		else:
			raise KeyError()

	def view(self, pointStart=None, pointEnd=None, featureStart=None,
			featureEnd=None):
		return self._source.view(pointStart, pointEnd, featureStart, featureEnd)

	####################################
	# Low Level Operations, Disallowed #
	####################################

	def setPointName(self, oldIdentifier, newName):
		"""
		Changes the pointName specified by previous to the supplied input name.
		
		oldIdentifier must be a non None string or integer, specifying either a current pointName
		or the index of a current pointName. newName may be either a string not currently
		in the pointName set, or None for an default pointName. newName cannot begin with the
		default prefix.

		None is always returned.

		"""
		self._readOnlyException("setPointName")

	def setFeatureName(self, oldIdentifier, newName):
		"""
		Changes the featureName specified by previous to the supplied input name.
		
		oldIdentifier must be a non None string or integer, specifying either a current featureName
		or the index of a current featureName. newName may be either a string not currently
		in the featureName set, or None for an default featureName. newName cannot begin with the
		default prefix.

		None is always returned.

		"""
		self._readOnlyException("setFeatureName")


	def setPointNames(self, assignments=None): 
		"""
		Rename all of the point names of this object according to the values
		specified by the assignments parameter. If given a list, then we use
		the mapping between names and array indices to define the point
		names. If given a dict, then that mapping will be used to define the
		point names. If assignments is None, then all point names will be
		given new default values. If assignment is an unexpected type, the names
		are not strings, the names are not unique, or point indices are missing,
		then an ArgumentException will be raised. None is always returned.

		"""
		self._readOnlyException("setPointNames")

	def setFeatureNames(self, assignments=None): 
		"""
		Rename all of the feature names of this object according to the values
		specified by the assignments parameter. If given a list, then we use
		the mapping between names and array indices to define the feature
		names. If given a dict, then that mapping will be used to define the
		feature names. If assignments is None, then all feature names will be
		given new default values. If assignment is an unexpected type, the names
		are not strings, the names are not unique, or feature indices are missing,
		then an ArgumentException will be raised. None is always returned.

		"""
		self._readOnlyException("setFeatureNames")


	###########################
	# Higher Order Operations #
	###########################
	
	def dropFeaturesContainingType(self, typeToDrop):
		"""
		Modify this object so that it no longer contains features which have the specified
		type as values. None is always returned.

		"""
		self._readOnlyException("dropFeaturesContainingType")

	def replaceFeatureWithBinaryFeatures(self, featureToReplace):
		"""
		Modify this object so that the chosen feature is removed, and binary valued
		features are added, one for each possible value seen in the original feature.
		None is always returned.

		"""		
		self._readOnlyException("replaceFeatureWithBinaryFeatures")

	def transformFeatureToIntegers(self, featureToConvert):
		"""
		Modify this object so that the chosen feature in removed, and a new integer
		valued feature is added with values 0 to n-1, one for each of n values present
		in the original feature. None is always returned.

		"""
		self._readOnlyException("transformFeatureToIntegers")

	def extractPointsByCoinToss(self, extractionProbability):
		"""
		Return a new object containing a randomly selected sample of points
		from this object, where a random experiment is performed for each
		point, with the chance of selection equal to the extractionProbabilty
		parameter. Those selected values are also removed from this object.

		"""
		self._readOnlyException("extractPointsByCoinToss")

	def shufflePoints(self, indices=None):
		"""
		Permute the indexing of the points so they are in a random order. Note: this relies on
		python's random.shuffle() so may not be sufficiently random for large number of points.
		See shuffle()'s documentation. None is always returned.

		"""
		self._readOnlyException("shufflePoints")
		
	def shuffleFeatures(self, indices=None):
		"""
		Permute the indexing of the features so they are in a random order. Note: this relies on
		python's random.shuffle() so may not be sufficiently random for large number of features.
		See shuffle()'s documentation. None is always returned.

		"""
		self._readOnlyException("shuffleFeatures")
		

	########################################
	########################################
	###   Functions related to logging   ###
	########################################
	########################################


	###############################################################
	###############################################################
	###   Subclass implemented information querying functions   ###
	###############################################################
	###############################################################


	##################################################################
	##################################################################
	###   Subclass implemented structural manipulation functions   ###
	##################################################################
	##################################################################

	def transpose(self):
		"""
		Function to transpose the data, ie invert the feature and point indices of the data.
		The point and feature names are also swapped. None is always returned.

		"""
		self._readOnlyException("transpose")

	def appendPoints(self, toAppend):
		"""
		Append the points from the toAppend object to the bottom of the features in this object.

		toAppend cannot be None, and must be a kind of data representation object with the same
		number of features as the calling object. None is always returned.
		
		"""
		self._readOnlyException("appendPoints")

	def appendFeatures(self, toAppend):
		"""
		Append the features from the toAppend object to right ends of the points in this object

		toAppend cannot be None, must be a kind of data representation object with the same
		number of points as the calling object, and must not share any feature names with the calling
		object. None is always returned.
		
		"""	
		self._readOnlyException("appendFeatures")

	def sortPoints(self, sortBy=None, sortHelper=None):
		""" 
		Modify this object so that the points are sorted in place, where sortBy may
		indicate the feature to sort by or None if the entire point is to be taken as a key,
		sortHelper may either be comparator, a scoring function, or None to indicate the natural
		ordering. None is always returned.
		"""
		self._readOnlyException("sortPoints")

	def sortFeatures(self, sortBy=None, sortHelper=None):
		""" 
		Modify this object so that the features are sorted in place, where sortBy may
		indicate the feature to sort by or None if the entire point is to be taken as a key,
		sortHelper may either be comparator, a scoring function, or None to indicate the natural
		ordering.  None is always returned.

		"""
		self._readOnlyException("sortFeatures")

	def extractPoints(self, toExtract=None, start=None, end=None, number=None, randomize=False):
		"""
		Modify this object, removing those points that are specified by the input, and returning
		an object containing those removed points.

		toExtract may be a single identifier, a list of identifiers, or a function that when
		given a point will return True if it is to be removed. number is the quantity of points that
		we are to be extracted, the default None means unlimited extraction. start and end are
		parameters indicating range based extraction: if range based extraction is employed,
		toExtract must be None, and vice versa. If only one of start and end are non-None, the
		other defaults to 0 and self.pointCount respectably. randomize indicates whether random
		sampling is to be used in conjunction with the number parameter, if randomize is False,
		the chosen points are determined by point order, otherwise it is uniform random across the
		space of possible removals.

		"""
		self._readOnlyException("extractPoints")

	def extractFeatures(self, toExtract=None, start=None, end=None, number=None, randomize=False):
		"""
		Modify this object, removing those features that are specified by the input, and returning
		an object containing those removed features. This particular function only does argument
		checking and modifying the featureNames for this object. It is the job of helper functions in
		the derived class to perform the removal and assign featureNames for the returned object.

		toExtract may be a single identifier, a list of identifiers, or a function that when
		given a feature will return True if it is to be removed. number is the quantity of features that
		are to be extracted, the default None means unlimited extraction. start and end are
		parameters indicating range based extraction: if range based extraction is employed,
		toExtract must be None, and vice versa. If only one of start and end are non-None, the
		other defaults to 0 and self.featureCount respectably. randomize indicates whether random
		sampling is to be used in conjunction with the number parameter, if randomize is False,
		the chosen features are determined by feature order, otherwise it is uniform random across the
		space of possible removals.

		"""
		self._readOnlyException("extractFeatures")

	def referenceDataFrom(self, other):
		"""
		Modifies the internal data of this object to refer to the same data as other. In other
		words, the data wrapped by both the self and other objects resides in the
		same place in memory. Other must be an object of the same type as
		the calling object. Also, the shape of other should be consistent with the set
		of featureNames currently in this object. None is always returned.

		"""
		self._readOnlyException("referenceDataFrom")

	def transformEachPoint(self, function, points=None):
		"""
		Modifies this object to contain the results of the given function
		calculated on the specified points in this object.

		function must not be none and accept the view of a point as an argument

		points may be None to indicate application to all points, a single point
		ID or a list of point IDs to limit application only to those specified.

		"""
		self._readOnlyException("transformEachPoint")


	def transformEachFeature(self, function, features=None):
		"""
		Modifies this object to contain the results of the given function
		calculated on the specified features in this object.

		function must not be none and accept the view of a feature as an argument

		features may be None to indicate application to all features, a single
		feature ID or a list of feature IDs to limit application only to those
		specified.

		"""
		self._readOnlyException("transformEachFeature")

	def transformEachElement(self, function, points=None, features=None, preserveZeros=False, skipNoneReturnValues=False):
		"""
		Modifies this object to contain the results of calling function(elementValue)
		or function(elementValue, pointNum, featureNum) for each element. 

		points: Limit to only elements of the specified points; may be None for
		all points, a single ID, or a list of IDs.

		features: Limit to only elements of the specified features; may be None for
		all features, a single ID, or a list of IDs.

		preserveZeros: If True it does not apply the function to elements in
		the data that are 0, and that 0 is not modified.

		skipNoneReturnValues: If True, any time function() returns None, the
		value originally in the data will remain unmodified.

		"""
		self._readOnlyException("transformEachElement")


	###############################################################
	###############################################################
	###   Subclass implemented numerical operation functions    ###
	###############################################################
	###############################################################

	def elementwiseMultiply(self, other):
		"""
		Perform element wise multiplication of this UML data object against the
		provided other UML data object, with the result being stored in-place in
		the calling object. Both objects must contain only numeric data. The
		pointCount and featureCount of both objects must be equal. The types of
		the two objects may be different. None is always returned.

		"""
		self._readOnlyException("elementwiseMultiply")

	def elementwisePower(self, other):
		self._readOnlyException("elementwisePower")

	def __imul__(self, other):
		"""
		Perform in place matrix multiplication or scalar multiplication, depending in the
		input 'other'

		"""
		self._readOnlyException("__imul__")

	def __iadd__(self, other):
		"""
		Perform in-place addition on this object, element wise if 'other' is a UML data
		object, or element wise with a scalar if other is some kind of numeric
		value.

		"""
		self._readOnlyException("__iadd__")

	def __isub__(self, other):
		"""
		Subtract (in place) from this object, element wise if 'other' is a UML data
		object, or element wise with a scalar if other is some kind of numeric
		value. 

		"""
		self._readOnlyException("__isub__")

	def __idiv__(self, other):
		"""
		Perform division (in place) using this object as the numerator, element
		wise if 'other' is a UML data object, or element wise by a scalar if other
		is some kind of numeric value.

		"""
		self._readOnlyException("__idiv__")

	def __itruediv__(self, other):
		"""
		Perform true division (in place) using this object as the numerator, element
		wise if 'other' is a UML data object, or element wise by a scalar if other
		is some kind of numeric value.

		"""
		self._readOnlyException("__itruediv__")

	def __ifloordiv__(self, other):
		"""
		Perform floor division (in place) using this object as the numerator, element
		wise if 'other' is a UML data object, or element wise by a scalar if other
		is some kind of numeric value.

		"""
		self._readOnlyException("__ifloordiv__")

	def __imod__(self, other):
		"""
		Perform mod (in place) using the elements of this object as the dividends,
		element wise if 'other' is a UML data object, or element wise by a scalar
		if other is some kind of numeric value.

		"""
		self._readOnlyException("__imod__")

	def __ipow__(self, other):
		"""
		Perform in-place exponentiation (iterated __mul__) using the elements
		of this object as the bases, element wise if 'other' is a UML data
		object, or element wise by a scalar if other is some kind of numeric
		value.

		"""
		self._readOnlyException("__ipow__")


	####################
	####################
	###   Helpers    ###
	####################
	####################

	def _readOnlyException(self, name):
		msg = "The " + name + " method is disallowed for View objects. View "
		msg += "objects are read only, yet this method modifies the object"
		raise ImproperActionException(msg)
