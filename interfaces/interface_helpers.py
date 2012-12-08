"""
Utility functions that could be useful in multiple interfaces

"""

import sys


def makeArgString(wanted, argDict, prefix, infix, postfix):
	"""
	Construct and return a long string containing argument and value pairs,
	each separated and surrounded by the given strings. If wanted is None,
	then no args are put in the string

	"""
	argString = ""
	if wanted is None:
		return argString
	for arg in wanted:
		if arg in argDict:
			value = argDict[arg]
			if isinstance(value,basestring):
				value = "\"" + value + "\""
			else:
				value = str(value)
			argString += prefix + arg + infix + value + postfix
	return argString


#TODO what about multiple levels???
def findModule(algorithmName, packageName, packageLocation):
	"""
	Import the desired python package, and search for the module containing
	the wanted algorithm. For use by interfaces to python packages.

	"""

	putOnSearchPath(packageLocation)
	exec("import " + packageName)

	contents = eval("dir(" + packageName + ")")

	# if all is defined, then we defer to the package to provide
	# a sensible list of what it contains
	if "__all__" in contents:
		contents = eval(packageName + ".__all__")

	for moduleName in contents:
		if moduleName.startswith("__"):
			continue
		cmd = "import " + packageName + "." + moduleName
		try:		
			exec(cmd)
		except ImportError as e:
			continue
		subContents = eval("dir(" + packageName + "." + moduleName + ")")
		if ".__all__" in subContents:
			contents = eval(packageName+ "." + moduleName + ".__all__")
		if algorithmName in subContents:
			return moduleName

	return None



def putOnSearchPath(wantedPath):
	if wantedPath is None:
		return
	elif wantedPath in sys.path:
		return
	else:
		sys.path.append(wantedPath)


#def representationBasedIOWrapper(backendToCall, trainData, testData, output, dependentVar, arguments):

	




