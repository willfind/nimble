"""
Contains the functions to be used for in-script calls to the external Regressors
package

Requirements: numpy
Possible requirements: scipy, matplotlib, svm

"""

import os
import os.path
import sys


try:
	import numpy
	loaded = True
except ImportError as ie:
	loaded = False
	message = ie


import UML
from UML.exceptions import ArgumentException

# Contains path to regressors root directory
regressorsDir = None

def setRegressorLocation(path):
	""" Sets the location of the root directory of the Regressor installation to be used """
	global regressorsDir
	regressorsDir = path


def getRegressorLocation():
	""" Returns the currently set path to the Regressor root directory """
	return regressorsDir



# TODO outer layer regressor() interface needs to deal with multiple kinds of input
def regressor(algorithm, trainData, testData, dependentVar=None, arguments={}, output=None, timer=None):
	"""
	Function to call on regressors in the Regression package.

	"""	
	if not loaded:
		print('Numpy must be installed in order for the regressors package to function')
		print ie
		return

	if not regressorsPresent():
		print('Cannot find the regressors package, please call setRegressorsLocation() with the ' +
				'path of the Regressors root directory')
		return

	if isinstance(testData, UML.data.Base):
		print('testData may not be an in package representation, it must refer to a file')
	
	if isinstance(trainData, UML.data.SparseData):
		raise ArgumentException("Regressors does not accept sparse input")

	testFile = open(testData, 'r')
	if output is not None:
		outFile = open(output, 'w')

	if not isinstance(trainData, UML.data.Base):
		trainData = UML.create("DenseMatrixData", data=trainData)
	
	# make sure dependentVar is a feature index
	if not isinstance(dependentVar, int):
		dependentVar = trainData.featureNames[dependentVar]
	
	# isolate the target values from training examples, if present
	trainDataY = None
	if dependentVar is not None:
		trainDataY = trainData.extractFeatures([dependentVar])
		# Regressors expects row vectors in this case
		trainDataY.transpose()

	# extract data and format data: Regressors expects numpy arrays
	trainData = numpy.array(trainData.data)
	if trainDataY is not None:
		trainDataY = numpy.array( trainDataY.data[0]).flatten()

	# import the type of regressor specified by the input
	try:
		sys.path.append(regressorsDir)
		cmd = "import " + algorithm
		exec(cmd)
	except ImportError as e:	
		print "regressors_interface failed to import " + algorithm
		print e
		return

	# determine the name of the regressor class in the file we've just imported
	algorithmClass = findRegressorClassName(regressorsDir + "/" + algorithm + ".py")
	if algorithmClass is None:
		print "Cannot find or determine a subclass of Regressor in " + algorithm
		return

	#start timing of classifier training, if timer is present
	if timer is not None:
		timer.start('train')

	#initialize the regressor with the constructed matrices
	cmd = algorithm + "." + algorithmClass + "(X=trainData,Y=trainDataY"
	for key in arguments.keys():
		cmd += "," + key + "="
		cmd += str(arguments[key])
	cmd += ")"
	regressor = eval(cmd)

	#stop timing of classifier training and start timing of testing if timer is present
	if timer is not None:
		timer.stop('train')
		timer.start('test')

	resultList = []
	# batch estimation, estimate() only takes a vector at a time
	for line in testFile:
		line = line.strip()
		lineList = line.split(",")
		if lineList == []:
			continue
		xVector = []
		for i in xrange(len(lineList)):
			xVector.append(float(lineList[i]))
	
		result = regressor.estimate(xVector)
		if output is None:
			resultList.append(result)
		else:
			outFile.write(str(result))
			outFile.write("\n")

	#stop timing of testing, if timer is present
	if timer is not None:
		timer.stop('test')
		
	if output is None:
		return UML.data.DenseMatrixData(resultList)


def regressorsPresent():
	""" Return true if we can establish that mahout is present """

	# check for nonesense values
	if regressorsDir is None or regressorsDir == '':
		return False

	# check that the path is to a valid directory
	if not os.path.isdir(regressorsDir):
		return False

	# that all we can do. If they set a directory, we just have to go with it
	return True


def findRegressorClassName(sourceFile):
	"""
	Given a python source file, returns the name of the first listed subclass of
	a class with 'Regressor' in the name, if such a class is present in the file.
	None otherwise
		
	"""
	toRead = open(sourceFile)

	for origLine in toRead:
		line = origLine.strip()
		if line.startswith("class "):
			splitLine = line.split("(")
			if len(splitLine) > 1 and "Regressor" in splitLine[1]:
				return splitLine[0][6:]

	return None


def listAlgorithms(includeParams=False):

	if not regressorsPresent():
		return []

	ret = []
	#look through each .py file in the directory
	contents = os.listdir(regressorsDir)
	for name in contents:
		# split from the right at most 1 times
		nameList = name.rsplit('.',1)
		if len(nameList) > 1 and nameList[1] == 'py':
			#look for class declaration
			className = findRegressorClassName(regressorsDir + "/" + name)
			if className is not None:
				ret.append(className)

	return ret







