"""
Module defining custom exceptions to be used in this package.
"""

class ArgumentException(Exception):
	"""
	Exception to be thrown when the value of an argument of some function
	renders it impossible to complete the operation that function is meant
	to perform.
	"""

	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)
	def __repr__(self):
		return "ArgumentException(" + repr(self.value) + ")"

class MissingEntryException(Exception):
	"""
	Exception to be thrown when a dictionary or other data structure is missing an entry
	that is necessary for the proper completion of an operation.
	"""
	
	def __init__(self, entryNames, value):
		self.entryNames = entryNames
		self.value = value

	def __str__(self):
		errMessage = "Missing entry names: " + self.entryNames[0]
		for entryName in self.entryNames[1:]:
			errMessage += ", " + entryName
		errMessage += '. \n' + repr(self.value)

class ImproperActionException(Exception):
	"""
		Exception to be thrown if calling a class function does not make sense in a
		certain context (i.e. calling call.turnOn() when the car object is already running),
		or is otherwise not allowed.
	"""

	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)

class EmptyFileException(Exception):
	"""
		Exception to be thrown if a file to be read is empty
	"""

	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)

class FileFormatException(Exception):
	"""
		Exception to be thrown if the formatting of a file is not
		as expected
	"""

	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)	

def prettyListString(inList, useAnd=False, numberItems=False, itemStr=str):
	ret = ""
	number = 0
	for i in xrange(len(inList)):
		value = inList[i]
		if i > 0:
			ret += ', '
			if useAnd and i == len(inList) - 1:
				ret += 'and '
		if numberItems:
			ret += '(' + str(number) + ') '
		ret += itemStr(value)
	return ret

def prettyDictString(inDict, useAnd=False, numberItems=False, keyStr=str, delim='=',
					valueStr=str):
	ret = ""
	number = 0
	keyList = inDict.keys()
	for i in xrange(len(keyList)):
		key = keyList[i]
		value = inDict[key]
		if i > 0:
			ret += ', '
			if useAnd and i == len(keyList) - 1:
				ret += 'and '
		if numberItems:
			ret += '(' + str(number) + ') '
		ret += keyStr(key) + delim + valueStr(value)
	return ret
