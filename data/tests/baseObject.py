
import UML
#import numpy

class DataTestObject(object):
	def __init__(self, returnType, constructor=None):
		def maker(data, pointNames=None, featureNames=None, name=None):
#			numP = len(pointNames) if pointNames is not None else 0
#			numF = len(featureNames) if featureNames is not None else 0
#			if data == None:
#				data = numpy.zeros(shape=(numP,numF))
			return UML.createData(returnType, data=data, pointNames=pointNames, featureNames=featureNames, name=name)
		if constructor is None:
			self.constructor = maker
		else:
			self.constructor = constructor
		self.returnType = returnType
