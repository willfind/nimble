
import UML
#import numpy

class DataTestObject(object):
	def __init__(self, returnType, constructor=None):
		def maker(data, pointNames=None, featureNames=None, name=None,
				path=(None,None)):
#			numP = len(pointNames) if pointNames is not None else 0
#			numF = len(featureNames) if featureNames is not None else 0
#			if data == None:
#				data = numpy.zeros(shape=(numP,numF))
			if isinstance(data, basestring):
				return UML.createData(returnType, data=data,
						pointNames=pointNames, featureNames=featureNames,
						name=name)
			else:
				return UML.helpers.initDataObject(returnType, rawData=data,
						pointNames=pointNames, featureNames=featureNames,
						name=name, path=path)
		if constructor is None:
			self.constructor = maker
		else:
			self.constructor = constructor
		self.returnType = returnType
