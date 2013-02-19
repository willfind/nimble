"""
Module defining custom exceptions to be used in this package.
"""

class ArgumentException (Exception):
	"""
	Exception to be thrown when the value of an argument of some function
	renders it impossible to complete the operation that function is meant
	to perform.
	"""

	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)

class MissingEntryException (Exception):
	def __init__(self, entryNames, value):
		self.entryNames = entryNames
		self.value = value
	def __str__(self):
		errMessage = "Missing entry names: " + self.entryNames[0]
		for entryName in self.entryNames[1:]:
			errMessage += ", " + entryName
		errMessage += '. \n' + repr(self.value)


