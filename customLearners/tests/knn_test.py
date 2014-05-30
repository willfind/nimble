
import UML

from nose.tools import *

def testKNNClassificationSimple():
	""" Test KNN classification by checking the ouput given simple hand made inputs """

	data = [[0,0,0], [1,10,10], [0,-1,4], [1,0,20]]
	trainObj = UML.createData('Matrix', data)

	data2 = [[2,2],[20,20]]
	testObj = UML.createData('Matrix', data2)

	name = 'Custom.KNNClassifier'
	ret = UML.trainAndApply(name, trainX=trainObj, trainY=0, testX=testObj, k=3)

	print ret.data

	assert ret[0,0] == 0
	assert ret[1,0] == 1

def testKNNClassificationSimpleScores():
	""" Test ridge regression by checking the shapes of the inputs and outputs """

	data = [[0,0,0], [1,10,10], [0,-1,4], [1,0,20]]
	trainObj = UML.createData('Matrix', data)

	data2 = [[2,2],[20,20]]
	testObj = UML.createData('Matrix', data2)

	name = 'Custom.KNNClassifier'
	tl = UML.train(name, trainX=trainObj, trainY=0, k=3)

	ret = tl.getScores(testObj, k=3)

	assert ret[0,0] == 2
	assert ret[0,1] == 1
	assert ret[1,0] == 1
	assert ret[1,1] == 2
