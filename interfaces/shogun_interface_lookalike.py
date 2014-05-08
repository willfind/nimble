"""




"""

import shogun_interface_old as shogun
from universal_interface_lookalike import UniversalInterfaceLookalike
from interface_helpers import cacheWrapper

class Shogun(UniversalInterfaceLookalike):
	"""

	"""

	def __init__(self):
		"""

		"""
		super(Shogun, self).__init__()

	def trainAndApply(self, learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', sendToLog=True):
		return shogun.shogun(learnerName, trainX, trainY, testX, arguments, output, scoreMode, 'default', sendToLog)


	@cacheWrapper
	def _listLearnersBackend(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		return shogun.listShogunLearners()

	def _getLearnerParameterNamesBackend(self, name):
		return shogun.getParameters(name)

	def _getLearnerDefaultValuesBackend(self, name):
		return shogun.getDefaultValues(name)

	def _getParameterNamesBackend(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		return shogun.getParameters(name)


	def _getDefaultValuesBackend(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""
		return shogun.getDefaultValues(name)


	def isAlias(self, name):
		"""
		Returns true if the name is an accepted alias for this interface

		"""
		if name.lower() in ['shogun']:
			return True
		else:
			return False


	def getCanonicalName(self):
		"""
		Returns the string name that will uniquely identify this interface

		"""
		return "shogun"

