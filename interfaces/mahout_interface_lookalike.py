"""




"""

import mahout_interface_old as mahout
from universal_interface_lookalike import UniversalInterfaceLookalike

class Mahout(UniversalInterfaceLookalike):
	"""

	"""

	def __init__(self):
		"""

		"""
		super(Mahout, self).__init__()

	def trainAndApply(self, learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', sendToLog=True):
		return mahout.mahout(learnerName, trainX, trainY, testX, arguments, output, sendToLog)


	def _listLearnersBackend(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		return mahout.listMahoutLearners()


	def _getParameterNamesBackend(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		return [[]]
		#return mahout.getParameters(name)

	def _getLearnerDefaultValuesBackend(self, name):
		return [[]]

	def _getLearnerParameterNamesBackend(self, name):
		return [[]]

	def _getDefaultValuesBackend(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""
		return [[]]
		#return mahout.getDefaultValues(name)

	def isAlias(self, name):
		"""
		Returns true if the name is an accepted alias for this interface

		"""
		if name.lower() in ['mahout']:
			return True
		else:
			return False


	def getCanonicalName(self):
		"""
		Returns the string name that will uniquely identify this interface

		"""
		return "mahout"
