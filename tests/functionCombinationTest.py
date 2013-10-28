
from UML import functionCombinations
from UML.umlHelpers import applyCodeVersions
from UML.umlHelpers import executeCode


TEXT_START = "\t\t\t------"
TEXT_END = "------"


def printCodeCombinations(functionText):
	combinations = functionCombinations(functionText)

	for number, combination in enumerate(combinations):
		print TEXT_START + " Version " + str(number+1) + " of " + str(len(combinations)) + TEXT_END
		print combination
		print ""


def printExample(functionText, variableshash):
	print ""
	print TEXT_START + " Starting code: " + TEXT_END + "\n" + functionText
	print ""
	print TEXT_START + " Variables " + TEXT_END + "\n" + str(variablesHash) + "\n\n"

	printCodeCombinations(functionText)

	results = applyCodeVersions(functionCombinations(functionText), variablesHash)

	print TEXT_START + " Results " + TEXT_END + "\n\n" + str(results) + "\n"


def copySort(x):
	import copy
	x = copy.copy(x)
	x.sort()
	return x


def testFunctionCombinations():
	"""nose test cases for the functionCombinations() function"""
	assert [4] == applyCodeVersions(functionCombinations("x+1"), {"x":3})
#	assert [2] == applyCodeVersions(functionCombinations("import math\nmath.sqrt(x)"), {"x":4})
	assert [5,6,7] == copySort(applyCodeVersions(functionCombinations("x+<1|2|3>"), {"x":4}))
	assert [5,6,7,10,11,12] == copySort(applyCodeVersions(functionCombinations("<x|y>+<1|2|3>"), {"x":4, "y":9}))
	assert [2,4,5,10] == copySort(applyCodeVersions(functionCombinations("z=<1|2.5>; <2|4>*z"), {}))
	assert [2,4,5,10] == copySort(applyCodeVersions(functionCombinations("z=<1|2.5>;w=<2|4>*z"), {}))


def testExecuteCode():
	"""nose test cases for the executeCode() function"""
	### setup a one-liner function
	vars = {"z":5, "y":9}
	f = "z+=3; y-=4; w=z+y; w+z"
	print "result:", executeCode(f, vars) 
	assert executeCode(f, vars) == 21
	### setup a full function definition as text
	g =\
"""
def f(z, y):
	z += 3
	y -= 4
	w = z+y
	return w+z
"""
	### define a python function object
	assert executeCode(g, vars) == 21
	def h(z, y):
		z += 3
		y -= 4
		w = z+y
		return w+z
	assert executeCode(h, vars) == 21









if __name__ == "__main__":
	print ""
	print "*"*80
	print "In this first example, we use a calculation case which we actually apply."
	print "*"*80

	variablesHash = {"x":3, "y":4}
	functionText = \
"""
def f(x,y):
	return <x|y>*y + <1|2|3>
"""

	printExample(functionText, variablesHash)


	print "\n"
	print "*"*80
	print "In this next example, we use a hypothetical machine learning case."
	print "*"*80
	print "\n"

	functionText = \
"""
def apply(X,Y):
	X.dropColumns([<0|3>])
	<X.normalizeStdDev()|X.normalizeMean()>
	return runAlgorithm(package="mlpy", learningAlgorithm="svm", parameters={"C":<0.1|1.0>})
"""

	print ""
	print TEXT_START + " Starting code: " + TEXT_END + "\n" + functionText
	print ""

	printCodeCombinations(functionText)


	print "\n"
	print "*"*80
	print "In this next example, we'll use a 'one liner'"
	print "*"*80
	print "\n"

	variablesHash = {"x":3, "y":4}
	functionText = "<x|y>*y + <1|2|3>"

	printExample(functionText, variablesHash)


	print "\n"
	print "*"*80
	print "In this next example, we'll use a 'one liner' that has multiple statements"
	print "*" * 80
	print "\n"

	variablesHash = {"x": 3, "y": 4}
	functionText = "z=x+<x|y>; <x|y>*z"

	printExample(functionText, variablesHash)





