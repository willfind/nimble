"""
Any method, object, or constant that might be used by multiple tests or
the main data wrapper objects defined in this module

"""


import copy
import math

from abc import ABCMeta
from abc import abstractmethod

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
		if not self.name().startswith(DEFAULT_PREFIX) and not other.name().startswith(DEFAULT_PREFIX):
			if self.name() != other.name():
				return False
		for i in xrange(len(self)):
			if self[i] != other[i]:
				return False
		return True

	def __str__(self):
		ret = '['
		for i in range(len(self)):
			if i != 0:
				ret +=', '
			ret += str(self[i])
		ret += ']'
		return ret

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


def _looksNumeric(val):
		if isinstance(val, basestring) or not hasattr(val, '__mul__'):
			return False
		return True

def formatIfNeeded(value, sigDigits):
	"""
	Format the value into a string, and in the case of a float typed value,
	limit the output to the given number of significant digits.

	"""
	if _looksNumeric(value):
		if int(value) != value and sigDigits is not None:
			return format(value, '.' + str(int(sigDigits)) + 'f')
	return str(value)

def indicesSplit(allowed, total):
	if total > allowed:
		allowed -= 1
	
	if allowed == 1 or total == 1:
		return ([0], [])

	forward = int(math.ceil(allowed/2.0))
	backward = int(math.floor(allowed/2.0))

	fIndices = range(forward)
	bIndices = range(-backward,0)

	for i in range(len(bIndices)):
		bIndices[i] = bIndices[i] + total

	if fIndices[len(fIndices)-1] == bIndices[0]:
		bIndices = bIndices[1:]

	return (fIndices, bIndices)


def makeNamesLines(indent, maxW, numDisplayNames, count, namesInv, nameType):
		namesString = ""
		(posL, posR) = indicesSplit(numDisplayNames, count)
		possibleIndices = posL + posR

		allDefault = True
		for i in range(len(possibleIndices)):
			if not namesInv[possibleIndices[i]].startswith(DEFAULT_PREFIX):
				allDefault = False

		if allDefault:
			return ""

		currNamesString = indent + nameType +'={'
		newStartString = indent * 2
		prevIndex = -1
		for i in range(len(possibleIndices)):
			currIndex = possibleIndices[i]
			# means there was a gap, need to insert elipses
			if currIndex - prevIndex > 1:
				addition = '..., '
				if len(currNamesString) + len(addition) > maxW:
					namesString += currNamesString + '\n'
					currNamesString = newStartString
				currNamesString += addition
			prevIndex = currIndex

			# get name and truncate if needed
			fullName = namesInv[currIndex]
			currName = fullName
			if len(currName) > 11:
				currName = currName[:8] + '...'
			addition = "'" + currName + "':" + str(currIndex)

			# if it isn't the last entry, comma and space. if it is
			# the end-cbrace
			addition += ', ' if i != (len(possibleIndices) -1) else '}'

			# if adding this would put us above the limit, add the line
			# to namesString before, and start a new line
			if len(currNamesString) + len(addition) > maxW:
				namesString += currNamesString + '\n'
				currNamesString = newStartString

			currNamesString += addition

		namesString += currNamesString + '\n'
		return namesString
