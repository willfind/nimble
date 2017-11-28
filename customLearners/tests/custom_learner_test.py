import UML

from nose.tools import *
import numpy.testing

from UML.customLearners import CustomLearner
from UML.configuration import configSafetyWrapper


@raises(TypeError)
def testCustomLearnerValidationNoType():
    """ Test  CustomLearner's validation for the learnerType class attribute """

    class NoType(CustomLearner):
        def train(self, trainX, trainY):
            return None

        def apply(self, testX):
            return None

    CustomLearner.validateSubclass(NoType)


@raises(TypeError)
def testCustomLearnerValidationWrongParamsTrain():
    """ Test CustomLearner's validation of required train() parameters """

    class WrongArgs(CustomLearner):
        learnerType = 'unknown'

        def train(self, trainZ, foo):
            return None

        def apply(self, testX):
            return None

    CustomLearner.validateSubclass(WrongArgs)


@raises(TypeError)
def testCustomLearnerValidationWrongParamsIncTrain():
    """ Test CustomLearner's validation of required incrementalTrain() parameters """

    class WrongArgs(CustomLearner):
        learnerType = 'unknown'

        def train(self, trainX, trainY):
            return None

        def incrementalTrain(self, trainZ, foo):
            return None

        def apply(self, testX):
            return None

    CustomLearner.validateSubclass(WrongArgs)


@raises(TypeError)
def testCustomLearnerValidationWrongParamsApply():
    """ Test CustomLearner's validation of required apply() parameters """

    class WrongArgs(CustomLearner):
        learnerType = 'unknown'

        def train(self, trainX, trainY):
            return None

        def apply(self, testZ):
            return None

    CustomLearner.validateSubclass(WrongArgs)


@raises(TypeError)
def testCustomLearnerValidationNoTrainOrIncTrain():
    """ Test CustomLearner's validation of requiring either train() or incrementalTrain() """

    class NoTrain(CustomLearner):
        learnerType = 'unknown'

        def apply(self, testX):
            return None

    CustomLearner.validateSubclass(NoTrain)


@raises(TypeError)
def testCustomLearnerValidationGetScoresParamsMatch():
    """ Test CustomLearner's validation of the match between getScores() param names and apply()"""

    class NoType(CustomLearner):
        learnerType = 'classification'

        def train(self, trainX, trainY):
            return None

        def apply(self, testX, foo):
            return None

        def getScores(self, testX):
            return None

    CustomLearner.validateSubclass(NoType)


@raises(TypeError)
def testCustomLearnerValidationInitNoParams():
    """ Test CustomLearner's validation of __init__'s params """

    class TooMany(CustomLearner):
        learnerType = 'classification'

        def __init__(self, so, many, params):
            super(TooMany, self).__init__()

        def train(self, trainX, trainY):
            return None

        def apply(self, testX, foo):
            return None

    CustomLearner.validateSubclass(TooMany)


@raises(TypeError)
def testCustomLearnerValidationInstantiates():
    """ Test CustomLearner's validation actually tries to instantiation the subclass """

    class NoApp(CustomLearner):
        learnerType = 'classification'

        def train(self, trainX, trainY):
            return None

    CustomLearner.validateSubclass(NoApp)


@raises(TypeError)
def testCustomLearnerValidationOptionsType():
    """ Test CustomLearner's validation the options() return type """

    class WrongOptType(CustomLearner):
        learnerType = 'classification'

        def __init__(self):
            super(WrongOptType, self).__init__()

        def train(self, trainX, trainY):
            return None

        def apply(self, testX, foo):
            return None

        @classmethod
        def options(self):
            return {}

    CustomLearner.validateSubclass(WrongOptType)


@raises(TypeError)
def testCustomLearnerValidationOptionsSubType():
    """ Test CustomLearner's validation of options() value type """

    class WrongOptSubType(CustomLearner):
        learnerType = 'classification'

        def __init__(self):
            super(WrongOptSubType, self).__init__()

        def train(self, trainX, trainY):
            return None

        def apply(self, testX, foo):
            return None

        @classmethod
        def options(self):
            return ['hello', 5]

    CustomLearner.validateSubclass(WrongOptSubType)


@raises(TypeError)
def testCustomLearnerValidationOptionsMethodOveride():
    """ Test CustomLearner's validation of options() value type """

    class WrongOptSubType(CustomLearner):
        learnerType = 'classification'

        def __init__(self):
            super(WrongOptSubType, self).__init__()

        def train(self, trainX, trainY):
            return None

        def apply(self, testX, foo):
            return None

        def options(self):
            return ['hello']

    CustomLearner.validateSubclass(WrongOptSubType)


class LoveAtFirstSightClassifier(CustomLearner):
    """ Always predicts the value of the first class it sees in the most recently trained data """
    learnerType = 'classification'

    def incrementalTrain(self, trainX, trainY):
        if hasattr(self, 'scope'):
            self.scope = numpy.union1d(self.scope, trainY.copyAs('numpyarray').flatten())
        else:
            self.scope = numpy.unique(trainY.copyAs('numpyarray'))
        self.prediction = trainY[0, 0]

    def apply(self, testX):
        ret = []
        for point in testX.pointIterator():
            ret.append([self.prediction])
        return UML.createData("Matrix", ret)

    def getScores(self, testX):
        ret = []
        for point in testX.pointIterator():
            currScores = []
            for value in self.scope:
                if value == self.prediction:
                    currScores.append(1)
                else:
                    currScores.append(0)
            ret.append(currScores)
        return UML.createData("Matrix", ret)


@configSafetyWrapper
def testCustomLearnerGetScores():
    """ Test that a CustomLearner with getScores can actually call that method """
    data = [[1, 3], [2, -5], [1, 44]]
    labels = [[0], [2], [1]]

    trainObj = UML.createData('Matrix', data)
    labelsObj = UML.createData('Matrix', labels)

    tdata = [[23, 2343], [23, 22], [454, -44]]
    testObj = UML.createData('Matrix', tdata)

    UML.registerCustomLearner("Custom", LoveAtFirstSightClassifier)

    name = 'Custom.LoveAtFirstSightClassifier'
    preds = UML.trainAndApply(name, trainX=trainObj, trainY=labelsObj, testX=testObj, scoreMode='label')
    assert preds.points == 3
    assert preds.features == 1
    best = UML.trainAndApply(name, trainX=trainObj, trainY=labelsObj, testX=testObj, scoreMode='bestScore')
    assert best.points == 3
    assert best.features == 2
    allScores = UML.trainAndApply(name, trainX=trainObj, trainY=labelsObj, testX=testObj, scoreMode='allScores')
    assert allScores.points == 3
    assert allScores.features == 3


@configSafetyWrapper
def testCustomLearnerIncTrainCheck():
    """ Test that a CustomLearner with incrementalTrain() but no train() works as expected """
    data = [[1, 3], [2, -5], [1, 44]]
    labels = [[0], [2], [1]]
    trainObj = UML.createData('Matrix', data)
    labelsObj = UML.createData('Matrix', labels)

    tdata = [[23, 2343], [23, 22], [454, -44]]
    testObj = UML.createData('Matrix', tdata)

    UML.registerCustomLearner("Custom", LoveAtFirstSightClassifier)

    def verifyScores(scores, currPredIndex):
        for rowNum in range(scores.points):
            for featNum in range(scores.features):
                value = scores[rowNum, featNum]
                if featNum == currPredIndex:
                    assert value == 1
                else:
                    assert value == 0

    name = 'Custom.LoveAtFirstSightClassifier'
    tlObj = UML.train(name, trainX=trainObj, trainY=labelsObj)

    origAllScores = tlObj.apply(testX=testObj, scoreMode='allScores')
    # label set [0,1,2] with 0 as constant prediction value
    verifyScores(origAllScores, 0)

    extendData = [[-343, -23]]
    extendLabels = [[3]]
    extTrainObj = UML.createData("Matrix", extendData)
    extLabelsObj = UML.createData("Matrix", extendLabels)

    tlObj.incrementalTrain(extTrainObj, extLabelsObj)

    incAllScores = tlObj.apply(testX=testObj, scoreMode='allScores')
    # label set now [0,1,2,3] with 3 as constant prediction value
    verifyScores(incAllScores, 3)

    reData = [[11, 12], [13, 14], [-22, -48]]
    reLabels = [[-1], [-1], [-2]]
    reTrainObj = UML.createData("Matrix", reData)
    reLabelsObj = UML.createData('Matrix', reLabels)

    tlObj.retrain(reTrainObj, reLabelsObj)
    reAllScores = tlObj.apply(testX=testObj, scoreMode='allScores')
    # label set now [-2,-1] with -1 as constant prediction value
    verifyScores(reAllScores, 1)
