
import UML

from UML.data.tests.numerical_backend import NumericalBackend
from UML.data.tests.query_backend import QueryBackend
from UML.data.tests.structure_backend import StructureBackend
from UML.data.tests.high_level_backend import HighLevelBackend
from UML.data.tests.low_level_backend import LowLevelBackend




class TestList(HighLevelBackend, NumericalBackend, QueryBackend, StructureBackend):
	def __init__(self):
		super(TestList, self).__init__('List')

class TestMatrix(HighLevelBackend, NumericalBackend, QueryBackend, StructureBackend):
	def __init__(self):
		super(TestMatrix, self).__init__('Matrix')

	def test_foldIterator_ordering(self):
		""" Test that foldIterator() yields folds in the proper order: X and Y folds should be in the same order"""
		twoColumnData = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]]
		matrixObj = UML.createData('Matrix', twoColumnData)
		Ydata = matrixObj.extractFeatures([1])
		Xdata = matrixObj
		XIterator = Xdata.foldIterator(numFolds=2)
		YIterator = Ydata.foldIterator(numFolds=2)
		
		while True: #need to add a test here for when iterator .next() is done
			try:
				curTrainX, curTestX = XIterator.next()
				curTrainY, curTestY = YIterator.next()
			except StopIteration:	#once we've gone through all the folds, this exception gets thrown and we're done!
				break
			curTrainXList = curTrainX.copyAs(format="python list")
			curTestXList = curTestX.copyAs(format="python list")
			curTrainYList = curTrainY.copyAs(format="python list")
			curTestYList = curTestY.copyAs(format="python list")

			for i in range(len(curTrainXList)):
				assert curTrainXList[i][0] == curTrainYList[i][0]

			for i in range(len(curTestXList)):
				assert curTestXList[i][0] == curTestYList[i][0]



class TestSparse(HighLevelBackend, NumericalBackend, QueryBackend, StructureBackend):
	def __init__(self):
		super(TestSparse, self).__init__('Sparse')


class TestBaseOnly(LowLevelBackend):
	def __init__(self):
		def makeConst(num):
			def const(dummy=2):
				return num
			return const
		def makeAndDefine(pointNames=None, featureNames=None, psize=0, fsize=0):			
			""" Make a base data object that will think it has as many features as it has featureNames,
			even though it has no actual data """
			rows = psize if pointNames is None else len(pointNames)
			cols = fsize if featureNames is None else len(featureNames)
			ret = UML.data.Base((rows,cols), pointNames=pointNames, featureNames=featureNames)
			return ret

		self.constructor = makeAndDefine
