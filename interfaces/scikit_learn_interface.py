"""
Contains the functions to be used for in-script calls to the scikit-learn
package

Relies on being scikit-learn 0.9 or above

"""

import inspect
import numpy

from interface_helpers import *
from ..processing.dense_matrix_data import loadCSV as DMDLoadCSV
from ..processing.dense_matrix_data import writeToCSV as DMDWriteToCSV
from ..processing.dense_matrix_data import DenseMatrixData as DMData
from ..processing.base_data import BaseData

from ..utility.custom_exceptions import ArgumentException

# Contains path to sciKitLearn root directory
sciKitLearnDir = None
locationCache = {}


def setSciKitLearnLocation(path):
	""" Sets the location of the root directory of the scikit-Learn installation to be used """
	global sciKitLearnDir
	sciKitLearnDir = path


def getSciKitLearnLocation():
	""" Returns the currently set path to the scikit-Learn root directory """
	return sciKitLearnDir
	

def sciKitLearnPresent():
	"""
	Return true if sciKitLearn is importable. If true, then the interface should
	be accessible.
	
	"""
	putOnSearchPath(sciKitLearnDir)
	try:
		import sklearn
	except ImportError:	
		return False

	return True


def sciKitLearn(algorithm, trainData, testData, output=None, dependentVar=None, arguments={}):
	"""
	Function to call on the estimator objects of the scikit-learn package.
	It will instantiate the estimator case-sensitively matching the given algorithm,
	using the trainData to in the .fit call, and testData in the .predict call,
	with all matching arguments from algArgs supplied at each step. If output
	is non-None, the output from the predict call is written to the supplied file,
	otherwise it is returned as a DenseMatrixData object.

	"""
	if not isinstance(trainData, BaseData):
		trainData = DMDLoadCSV(trainData)
	if not isinstance(testData, BaseData):
		testData = DMDLoadCSV(testData)

	trainDataY = None
	# directly assign target values, if present
	if isinstance(dependentVar, BaseData):
		trainDataY = dependentVar
	# isolate the target values from training examples, otherwise
	elif dependentVar is not None:
		trainDataY = trainData.extractColumns([dependentVar])		
		trainDataY = trainDataY.convertToDenseMatrixData()
		trainDataY.transpose()
		
	# extract data
	trainData = trainData.data
	if trainDataY is not None:
		#trainDataY = trainDataY.data
		toFlatten = trainDataY.data
		if not isinstance(dependentVar, BaseData):
			toFlatten = toFlatten[0]
		trainDataY = numpy.array(toFlatten).flatten()
	testData = testData.data

	# call backend
	try:
		retData = _sciKitLearnBackend(algorithm, trainData, trainDataY, testData, arguments)
	except ImportError as e:
		print "ImportError: " + str(e)
		if not sciKitLearnPresent():
			print "Scikit-learn not importable."
			print "It must be either on the search path, or have its path set by setSciKitLearnLocation()"
		return

	if retData is None:
		return

	outputObj = DMData(retData)

	if output is None:
		return outputObj

	DMDWriteToCSV(outputObj,output,False)


def _sciKitLearnBackend(algorithm, trainDataX, trainDataY, testData, algArgs):
	"""
	Function to find, construct, and execute the wanted calls to scikit-learn

	"""	
	global locationCache
	if algorithm in locationCache:
		moduleName = locationCache[algorithm]
	else: 
		moduleName = findModule(algorithm, "sklearn", sciKitLearnDir)		
		locationCache[algorithm] = moduleName

	if moduleName is None:
		print ("Could not find the algorithm")
		return

	putOnSearchPath(sciKitLearnDir)
	exec ("from sklearn import " + moduleName)

	# make object
	objectCall = moduleName + "." + algorithm
	(objArgs,v,k,d) = eval("inspect.getargspec(" + objectCall + ".__init__)")
	argString = makeArgString(objArgs, algArgs, "", "=", ", ")
	sklObj = eval(objectCall + "(" + argString + ")")

	# fit object
	(fitArgs,v,k,d) = inspect.getargspec(sklObj.fit)
	argString = makeArgString(fitArgs, algArgs, "", "=", ", ")
	sklObj = eval("sklObj.fit(trainDataX,trainDataY " + argString + ")")

	# estimate from object
	(preArgs,v,k,d) = inspect.getargspec(sklObj.predict)
	argString = makeArgString(preArgs, algArgs, "", "=", ", ")
	outData = eval("sklObj.predict(testData, " + argString + ")")

	return outData



def listAlgorithms():
	"""
	Function to return a list of all algorithms callable through our interface, if scikit learn is present
	
	"""
	if not sciKitLearnPresent():
		return []

	import sklearn

	ret = []
	subpackages = sklearn.__all__

	for sub in subpackages:
		curr = 'sklearn.' + sub
		try:
			exec('import ' + curr)
		except ImportError:
			# no guarantee __all__ is accurate, if something doesn't import, just skip ahead
			continue

		contents = eval('dir(' + curr + ')')
		
		for member in contents:
			memberContents = eval('dir(' + curr + "." + member + ')')
			if 'fit' in memberContents and 'predict' in memberContents:
				ret.append(member)

	return ret


