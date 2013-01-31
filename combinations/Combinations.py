"""
Problem:
-Find a way to represent arbitrary sequences of function calls that occur one after another
-Find a way to generate combinations of these sequences that have all combinations of various parameters

Representation:

"""

import re  # regular expression library
import math

from UML import *

TEXT_START = "\t\t\t------"
TEXT_END = "------"

def functionCombinations(functionText):
	"""process the text of the given python code, returning a list of different versions of the text. Any occurence of, say, <41|6|3>
		generate three different versions of the code, one with 41, one with 6, one with 3. This notation <...|...|...> can have any number of vertical bars
		and can be used anywhere in the text. The ... portions can be anything."""
	resultsSoFar = [functionText]	#as we process the <...|...|...> patterns one at a time, we keep adding all versions of the text so far to this list.
	while True:	#we'll keep searching the text until we can't find a <...|...|...> pattern
		newResults = []	#our next round of results that we'll generate in a moment
		done = False
		for text in resultsSoFar:
			result = re.search("(<[^\|]+(?:\|[^\|>]+)*>)", text) #an incomprehensable regular expression that searches for strings of the form <...|...|...>
			if result == None: #no more occurences, so we're done
				done = True
				break
			start = result.start()+1
			end = result.end()-1
			pieces = text[start:end].split("|")
			for piece in pieces:
				newText = text[:start-1] + str(piece) + text[end+1:]
				newResults.append(newText)
		if not done: resultsSoFar = newResults
		if done: break
	return resultsSoFar


def applyCodeVersions(functionTextList, inputHash):
	"""applies all the different various versions of code that can be generated from functionText to each of the variables specified in inputHash, where
	data is plugged into the variable with name inputVariableName. Returns the result of each application as a list.
	functionTextList is a list of text objects, each of which defines a python function
	inputHash is of the form {variable1Name:variable1Value, variable2Name:variable2Value, ...}
	"""
	results = []
	for codeText in functionTextList:
		results.append(executeCode(codeText, inputHash))
	return results


def executeCode(code, inputHash):
	"""Execute the given code stored as text in codeText, starting with the variable values specified in inputHash
	This function assumes the code consists of EITHER an entire function definition OR a single line of code with
	statements seperated by semi-colons OR as a python function object.
	"""
	#inputHash = inputHash.copy() #make a copy so we don't modify it... but it doesn't seem necessary
	if isSingleLineOfCode(code): return executeOneLinerCode(code, inputHash) #it's one line of text (with ;'s to seperate statemetns')
	elif isinstance(code, (str, unicode)): return executeFunctionCode(code, inputHash) #it's the text of a function definition
	else: return code(**inputHash)	#assume it's a function itself


def executeOneLinerCode(codeText, inputHash):
	"""Execute the given code stored as text in codeText, starting with the variable values specified in inputHash
	This function assumes the code consists of just one line (with multiple statements seperated by semi-colans.
	Note: if the last statement in the line starts X=... then the X= gets stripped off (to prevent it from getting broken by A=(X=...)).
	"""
	if not isSingleLineOfCode(codeText): raise Exception("The code text was not just one line of code.")
	codeText = codeText.strip()
	localVariables = inputHash.copy()
	pieces = codeText.split(";")
	lastPiece = pieces[-1].strip()
	lastPiece = re.sub("\A([\w])+[\s]*=", "",  lastPiece) #if the last statement begins with something like X = ... this removes the X = part.
	lastPiece = lastPiece.strip()
	pieces[-1] = "RESULTING_VALUE_ZX7_ = (" + lastPiece + ")"
	codeText = ";".join(pieces)
	#oneLiner = True

	exec(codeText, globals(), localVariables)	#apply the code
	return localVariables["RESULTING_VALUE_ZX7_"]



def executeFunctionCode(codeText, inputHash):
	"""Execute the given code stored as text in codeText, starting with the variable values specified in inputHash
	This function assumes the code consists of an entire function definition.
	"""
	if not "def" in codeText: raise Exception("No function definition was found in this code!")
	localVariables = {}
	exec(codeText, globals(), localVariables)	#apply the code, which declares the function definition
	#foundFunc = False
	#result = None
	for varName, varValue in localVariables.iteritems():
		if "function" in str(type(varValue)):
			return varValue(**inputHash)
	

def isSingleLineOfCode(codeText):
	if not isinstance(codeText, (str, unicode)): return False
	codeText = codeText.strip()
	try:
		codeText.strip().index("\n")
		return False
	except ValueError:
		return True



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

def sort(x):
	import copy
	x = copy.copy(x)
	x.sort()
	return x


def testFunctionCombinations():
	"""nose test cases for the functionCombinations() function"""
	assert [4] == applyCodeVersions(functionCombinations("x+1"), {"x":3})
	assert [2] == applyCodeVersions(functionCombinations("math.sqrt(x)"), {"x":4})
	assert [5,6,7] == sort(applyCodeVersions(functionCombinations("x+<1|2|3>"), {"x":4}))
	assert [5,6,7,10,11,12] == sort(applyCodeVersions(functionCombinations("<x|y>+<1|2|3>"), {"x":4, "y":9}))
	assert [2,4,5,10] == sort(applyCodeVersions(functionCombinations("z=<1|2.5>; <2|4>*z"), {}))
	assert [2,4,5,10] == sort(applyCodeVersions(functionCombinations("z=<1|2.5>;w=<2|4>*z"), {}))


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
	return runAlgorithm(package="mlpy", algorithm="svm", parameters={"C":<0.1|1.0>})
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







