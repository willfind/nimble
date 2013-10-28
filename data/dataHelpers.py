"""
Any method, object, or constant that might be used by multiple tests or
the main data wrapper objects defined in this module

"""


import copy
from abc import ABCMeta
from abc import abstractmethod


# a default seed for testing and predictible trials
DEFAULT_SEED = 'DEFAULTSEED'

# the prefix for default featureNames
DEFAULT_PREFIX = "_DEFAULT_#"

DEFAULT_NAME_PREFIX = "OBJECT_#"

defaultObjectNumber = 0


class View():
	__metaclass__ = ABCMeta

	def equals(self, other):
		if not isinstance(other, View):
			return False
		if len(self) != len(other):
			return False
		if self.index() != other.index():
			return False
		if self.name() != other.name():
			return False
		for i in xrange(len(self)):
			if self[i] != other[i]:
				return False
		return True

	@abstractmethod
	def __getitem__(self, index):
		pass
	@abstractmethod
	def __setitem__(self, key, value):
		pass
	@abstractmethod
	def nonZeroIterator(self):
		pass
	@abstractmethod
	def __len__(self):
		pass
	@abstractmethod
	def index(self):
		pass
	@abstractmethod
	def name(self):
		pass



def reorderToMatchExtractionList(dataObject, extractionList, axis):
	"""
	Helper which will reorder the data object along the specified axis so that
	instead of being in an order corresponding to a sorted version of extractionList,
	it will be in the order of the given extractionList.

	extractionList must contain only indices, not name based identifiers.
	
	"""
	if axis.lower() == "point":
		sortFunc = dataObject.sortPoints
	else:
		sortFunc = dataObject.sortFeatures

	sortedList = copy.copy(extractionList)
	sortedList.sort()
	mappedOrig = {}
	for i in xrange(len(extractionList)):
		mappedOrig[extractionList[i]] = i
	
	def scorer(viewObj):
		return mappedOrig[sortedList[viewObj.index()]]

	sortFunc(sortHelper=scorer)

	return dataObject




