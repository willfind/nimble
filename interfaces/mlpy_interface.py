"""
Contains the functions to be used for in-script calls to the mlpy
package

"""

import inspect
import numpy

from interface_helpers import *
from ..processing.dense_matrix_data import DenseMatrixData as DMData
from ..processing.base_data import BaseData
from ..processing.sparse_data import SparseData

# Contains path to mlpy root directory
mlpyDir = None
locationCache = {}


def setMlpyLocation(path):
	""" Sets the location of the root directory of the Regressor installation to be used """
	global mlpyDir
	mlpyDir = path


def getMlpyLocation():
	""" Returns the currently set path to the  root directory """
	return mlpyDir
	

def mlpyPresent():
	"""
	Return true if mlpy is importable. If true, then the interface should
	be accessible.
	
	"""
	putOnSearchPath(mlpyDir)
	try:
		import mlpy
	except ImportError:	
		return False

	return True


def mlpy(algorithm, trainData, testData, output=None, dependentVar=None, arguments={}):
	"""


	"""
	if isinstance(trainData, SparseData):
		raise ArgumentException("MLPY does not accept sparse input")
	if isinstance(testData, SparseData):
		raise ArgumentException("MLPY does not accept sparse input")

	if not isinstance(trainData, BaseData):
		trainObj = DMData(file=trainData)
	else: # input is an object
		trainObj = trainData
	if not isinstance(testData, BaseData):
		testObj = DMData(file=testData)
	else: # input is an object
		testObj = testData
	
	trainObjY = None
	# directly assign target values, if present
	if isinstance(dependentVar, BaseData):
		trainObjY = dependentVar
	# otherwise, isolate the target values from training examples
	elif dependentVar is not None:
		# TODO currently destructive!
		trainObjY = trainObj.extractFeatures([dependentVar])		
	# could be None for unsupervised learning	

	# necessary format for skl, also makes the following ops easier
	if trainObjY is not None:	
		trainObjY = trainObjY.toDenseMatrixData()
	
	# pull out data from obj
	trainRawData = trainObj.data
	if trainObjY is not None:
		# corrects the dimensions of the matrix data to be just an array
		trainRawDataY = numpy.array(trainObjY.data).flatten()
	else:
		trainRawDataY = None
	testRawData = testObj.data

	# call backend
	try:
		retData = _mlpyBackend(algorithm, trainRawData, trainRawDataY, testRawData, arguments)
	except ImportError as e:
		print "ImportError: " + str(e)
		if not mlpyPresent():
			print "mlpy not importable."
			print "It must be either on the search path, or have its path set by setMlpyLocation()"
		return

	if retData is None:
		return

	outputObj = DMData(retData)

	if output is None:
		return outputObj

	outputObj.writeCSV(output,False)


def _mlpyBackend(algorithm, trainDataX, trainDataY, testData, algArgs):
	"""
	Function to find, construct, and execute the wanted calls to mlpy

	"""	
	putOnSearchPath(mlpyDir)
	import mlpy 

	# Deal with kernel transformations
	kernObj = None
	if 'kernel' in algArgs and algArgs['kernel'] is not None:
		kernelName = algArgs['kernel']
		objectCall = "mlpy." + kernelName
		try:
			(kernArgs,v,k,d) = eval("inspect.getargspec(" + objectCall + ".__init__)")
		except TypeError:
			#default to nothing
			kernArgs = {}
		argString = makeArgString(kernArgs, algArgs, "", "=", ", ")
		kernObj = eval(objectCall + "(" + argString + ")")
		del algArgs['kernel']

	# make object
	objectCall = "mlpy." + algorithm
	try:
		(objArgs,v,k,d) = eval("inspect.getargspec(" + objectCall + ".__init__)")
	except TypeError:
		# this can occur if we are inspecting a C backed function. We default
		# to adding everything available.
		objArgs = algArgs
	argString = makeArgString(objArgs, algArgs, "", "=", ", ")
	if kernObj is not None:
		argString += "kernel=kernObj" 
	obj = eval(objectCall + "(" + argString + ")")

	# run code for the learn / pred paradigm
	if hasattr(obj, 'pred'):
		# call .learn for the object
		try:
			(learnArgs,v,k,d) = inspect.getargspec(obj.learn)
		except TypeError:
			# in this case, we default to adding nothing
			learnArgs = None
		argString = makeArgString(learnArgs, algArgs, "", "=", ", ")
		eval("obj.learn(trainDataX,trainDataY " + argString + ")")

		# call .pred for the object
		try:
			(predArgs,v,k,d) = inspect.getargspec(obj.pred)
		except TypeError:
			# in this case, we default to adding nothing
			predArgs = None
		argString = makeArgString(predArgs, algArgs, "", "=", ", ")
		outData = eval("obj.pred(testData, " + argString + ")")
		# .learn() always returns a row vector, we want a column vector
		outData.resize(outData.size,1)

	# run code for the learn / transform paradigm
	if hasattr(obj, 'transform'):
		# call .learn for the object
		try:
			(learnArgs,v,k,d) = inspect.getargspec(obj.learn)
		except TypeError:
			# in this case, we default to adding nothing
			learnArgs = None
		argString = makeArgString(learnArgs, algArgs, "", "=", ", ")
		eval("obj.learn(trainDataX, " + argString + ")")

		# call .transform for the object
		try:
			(transArgs,v,k,d) = inspect.getargspec(obj.transform)
		except TypeError:
			# in this case, we default to adding nothing
			transArgs = None
		argString = makeArgString(transArgs, algArgs, "", "=", ", ")
		outData = eval("obj.transform(testData, " + argString + ")")

	return outData



def listAlgorithms(includeParams=False):
	"""
	Function to return a list of all algorithms callable through our interface, if mlpy is present
	
	"""
	if not mlpyPresent():
		return []

	import mlpy

	ret = []
	contents = dir(mlpy)

	for member in contents:
		subContents = eval('dir(mlpy.' + member + ')')
		if 'learn' in subContents and ('pred' in subContents or 'transform' in subContents):
			if not includeParams:
				ret.append(member)
			else:
				try:
					(objArgs,v,k,d) = eval("inspect.getargspec(" + objectCall + ".__init__)")
					(learnArgs,v,k,d) = inspect.getargspec(obj.learn)
					if 'pred' in subContents:
						(lastArgs,v,k,d) = inspect.getargspec(obj.pred)
					else:		
						(lastArgs,v,k,d) = inspect.getargspec(obj.transform)
				except TypeError:
					pass
				

	return ret


