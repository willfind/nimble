
import UML
#import numpy

class DataTestObject(object):
	def __init__(self, retType):
		def maker(data, pointNames=None, featureNames=None):
#			numP = len(pointNames) if pointNames is not None else 0
#			numF = len(featureNames) if featureNames is not None else 0
#			if data == None:
#				data = numpy.zeros(shape=(numP,numF))
			return UML.createData(retType, data=data, featureNames=featureNames)
		self.constructor = maker
