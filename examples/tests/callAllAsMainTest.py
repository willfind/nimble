"""
Defines a single test to check the functionality of all of the
scripts contained in the above folder.



"""

import os
import sys

#ensures UML.examples.allowImports is in sys.modules
import UML.examples.allowImports


def test_callAllAsMain():
	"""
	Test which calls each script in the examples folder individually, and
	confirms that it does not throw any sort of exception.

	"""

	# bind an the name allowImports to the appropriate, already loaded, module
	# needed because each script we call imports a function from allowImports,
	# but since we are calling from a different context, the import as given
	# fails.
	sys.modules['allowImports'] = sys.modules['UML.examples.allowImports']

	cwd = os.getcwd()
	parent = os.path.dirname(cwd)

	# this just makes the calls easier
	os.chdir(parent)

	examples = os.listdir(parent)
#	ladderExamples = os.listdir(os.path.join(parent,"laddersExperiments"))
#	for x in ladderExamples:
#		examples.append(os.path.join("laddersExperiments", x))

	cleaned = []
	for fileName in examples:
		if fileName.endswith('.py') and not fileName.startswith("__"):
			cleaned.append(fileName)

	# so when we execute the scripts, we actually run the poriton that is meant
	# to be run when called as main
	__name__ = "__main__"

	results = {}

	for script in cleaned:
		results[script] = None
		try:
			execfile(script)
		except Exception as e:
			results[script] = e

	print ""
	print "*** Results ***"
	print ""
	sortedKeys = sorted(results.keys())
	for key in sortedKeys:
		print key +" : " + str(results[key])
		assert results[key] is None

#	assert False
