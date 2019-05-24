"""
Unit tests for keras_interface.py

Cannot test for expected result equality due to inability to control for
randomness in Keras.
"""

from __future__ import absolute_import

import numpy as np
from nose.tools import raises

import UML as nimble
from UML import createData
from UML.interfaces.keras_interface import Keras
from UML.exceptions import InvalidArgumentValue
from .skipTestDecorator import SkipMissing
from ..assertionHelpers import logCountAssertionFactory, noLogEntryExpected

keras = nimble.importExternalLibraries.importModule("keras")

keraSkipDec = SkipMissing('Keras')

@keraSkipDec
@noLogEntryExpected
def test_Keras_version():
    interface = Keras()
    assert interface.version() == keras.__version__

@keraSkipDec
@logCountAssertionFactory(5)
def testKerasAPI():
    """
    Test Keras can handle a variety of arguments passed to all major learning functions
    """
    x_train = createData('Matrix', np.random.random((1000, 20)), useLog=False)
    y_train = createData('Matrix', np.random.randint(2, size=(1000, 1)), useLog=False)

    layer0 = {'type':'Dense', 'units':64, 'activation':'relu', 'input_dim':20}
    layer1 = {'type':'Dropout', 'rate':0.5}
    layer2 = {'type':'Dense', 'units':1, 'activation':'sigmoid'}
    layers = [layer0, layer1, layer2]

    #####test fit
    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer='sgd',
                       layers=layers, loss='binary_crossentropy', metrics=['accuracy'],
                       epochs=20, batch_size=128, lr=0.1, decay=1e-6, momentum=0.9,
                       nesterov=True)

    #######test apply
    x = mym.apply(testX=x_train)

    #########test trainAndApply
    x = nimble.trainAndApply('keras.Sequential', trainX=x_train, trainY=y_train, testX=x_train,
                             optimizer='sgd', layers=layers, loss='binary_crossentropy',
                             metrics=['accuracy'], epochs=20, batch_size=128, lr=0.1,
                            decay=1e-6, momentum=0.9, nesterov=True)

    #########test trainAndTest
    x = nimble.trainAndTest('keras.Sequential', trainX=x_train, testX=x_train, trainY=y_train,
                            testY=y_train, optimizer='sgd', layers=layers,
                            loss='binary_crossentropy', metrics=['accuracy'], epochs=20,
                            batch_size=128, lr=0.1, decay=1e-6, momentum=0.9, nesterov=True,
                            performanceFunction=nimble.calculate.loss.rootMeanSquareError)

    #########test CV
    results = nimble.crossValidate(
        "keras.Sequential", X=x_train, Y=y_train, optimizer='sgd', layers=layers,
        loss='binary_crossentropy', metrics=['accuracy'], epochs=20, batch_size=128,
        lr=0.1, decay=1e-6, momentum=0.9, nesterov=True,
        performanceFunction=nimble.calculate.loss.rootMeanSquareError)
    bestArguments = results.bestArguments
    bestScore = results.bestResult

@keraSkipDec
@logCountAssertionFactory(3)
def testKerasIncremental():
    """
    Test Keras can handle and incrementalTrain call
    """
    x_train = createData('Matrix', np.random.random((1000, 20)), useLog=False)
    y_train = createData('Matrix', np.random.randint(2, size=(1000, 1)), useLog=False)

    layer0 = {'type':'Dense', 'units':64, 'activation':'relu', 'input_dim':20}
    layer1 = {'type':'Dropout', 'rate':0.5}
    layer2 = {'type':'Dense', 'units':1, 'activation':'sigmoid'}
    layers = [layer0, layer1, layer2]

    #####test fit
    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer='sgd',
                       layers=layers, loss='binary_crossentropy', metrics=['accuracy'],
                       epochs=20, batch_size=128, lr=0.1, decay=1e-6, momentum=0.9,
                       nesterov=True)

    #####test incrementalTrain
    mym.incrementalTrain(x_train, y_train)

    x = mym.apply(testX=x_train)
    assert len(x.points) == len(x_train.points)

@keraSkipDec
@logCountAssertionFactory(2)
def testKeras_Sparse_FitGenerator():
    """
    Test Keras on Sparse data; uses different training method - fit_generator
    """
    x_data = np.random.random((20, 7))
    y_data = np.random.randint(2, size=(20, 1))

    x_train = createData('Sparse', x_data, useLog=False)
    y_train = createData('Matrix', y_data, useLog=False)

    layer0 = {'type':'Dense', 'units':64, 'activation':'relu', 'input_dim':7}
    layer1 = {'type':'Dropout', 'rate':0.5}
    layer2 = {'type':'Dense', 'units':1, 'activation':'sigmoid'}
    layers = [layer0, layer1, layer2]

    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer='sgd',
                       layers=layers, loss='binary_crossentropy', metrics=['accuracy'],
                       epochs=2, batch_size=1, lr=0.1, decay=1e-6, momentum=0.9, nesterov=True,
                       steps_per_epoch=20, max_queue_size=1, shuffle=False, steps=20)

    x = mym.apply(testX=x_train)
    assert len(x.points) == len(x_train.points)

@keraSkipDec
@logCountAssertionFactory(3)
def test_TrainedLearnerApplyArguments():
    """ Test a keras function that accept arguments for predict"""
    x_train = createData('Matrix', np.random.random((1000, 20)), useLog=False)
    y_train = createData('Matrix', np.random.randint(2, size=(1000, 1)), useLog=False)

    layer0 = {'type':'Dense', 'units':64, 'activation':'relu', 'input_dim':20}
    layer1 = {'type':'Dropout', 'rate':0.5}
    layer2 = {'type':'Dense', 'units':1, 'activation':'sigmoid'}
    layers = [layer0, layer1, layer2]

    # Sequential.predict takes a 'steps' argument. Default is 20
    mym = nimble.train(
        'keras.Sequential', trainX=x_train, trainY=y_train,
        optimizer='sgd', layers=layers, loss='binary_crossentropy',
        metrics=['accuracy'], epochs=1, lr=0.1, decay=1e-6, momentum=0.9,
        nesterov=True, shuffle=False)
    # using arguments parameter
    newArgs = mym.apply(testX=x_train, arguments={'steps': 50})
    # using kwarguments
    newArgs = mym.apply(testX=x_train, steps=50)

@keraSkipDec
@logCountAssertionFactory(1)
def test_TrainedLearnerApplyArguments_exception():
    """ Test an keras function with invalid arguments for predict"""
    x_train = createData('Matrix', np.random.random((1000, 20)), useLog=False)
    y_train = createData('Matrix', np.random.randint(2, size=(1000, 1)), useLog=False)

    layer0 = {'type':'Dense', 'units':64, 'activation':'relu', 'input_dim':20}
    layer1 = {'type':'Dropout', 'rate':0.5}
    layer2 = {'type':'Dense', 'units':1, 'activation':'sigmoid'}
    layers = [layer0, layer1, layer2]

    # Sequential.predict does not take a 'foo' argument.
    mym = nimble.train(
        'keras.Sequential', trainX=x_train, trainY=y_train,
        optimizer='sgd', layers=layers, loss='binary_crossentropy',
        metrics=['accuracy'], epochs=1, lr=0.1, decay=1e-6, momentum=0.9,
        nesterov=True, shuffle=False)
    try:
        # using arguments parameter
        newArgs1 = mym.apply(testX=x_train, arguments={'foo': 50})
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
    try:
        # using kwarguments
        newArgs2 = mym.apply(testX=x_train, foo=50)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
