"""
Defines a single test to check the functionality of all of the
scripts contained in the above folder.



"""

import os
import sys

import UML
#ensures UML.examples.allowImports is in sys.modules
import UML.examples.allowImports


def test_callAllAsMain():
	"""
	Calls each script in examples, confirms complete with an exception.

	"""
	assert False  # Disabled until examples are all fixed.

	# bind the name allowImports to the appropriate, already loaded, module
	# needed because each script we call imports a function from allowImports,
	# but since we are calling from a different context, the import as given
	# fails.
	sys.modules['allowImports'] = sys.modules['UML.examples.allowImports']

	examplesDir = os.path.join(UML.UMLPath, 'examples')

	examplesFiles = os.listdir(examplesDir)

	cleaned = []
	for fileName in examplesFiles:
		if fileName.endswith('.py') and not fileName.startswith("__"):
			cleaned.append(fileName)

	# so when we execute the scripts, we actually run the poriton that is meant
	# to be run when called as main
	__name__ = "__main__"

	results = {}

	for script in cleaned:
		results[script] = None
		try:
			execfile(os.path.join(examplesDir, script))
		except Exception as e:
			results[script] = (e, sys.exc_info())

	print ""
	print "*** Results ***"
	print ""
	print ""
	sortedKeys = sorted(results.keys())
	for key in sortedKeys:
		val = results[key]
		print key + " : " + str(val)
		print ""
		#if isinstance(val, tuple) and len(val) > 0 and isinstance(val[0], Exception):
			#raise val[1][1], None, val[1][2]
			#print key
			#print val[1][1], None, val[1][2]
