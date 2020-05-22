"""
Unit tests for keras_interface.py

Cannot test for expected result equality due to inability to control for
randomness in Keras.
"""

import numpy as np
from nose.tools import raises

import nimble
from nimble import createData
from nimble.interfaces.keras_interface import Keras
from nimble.exceptions import InvalidArgumentValue
from .skipTestDecorator import SkipMissing
from ..assertionHelpers import logCountAssertionFactory, noLogEntryExpected

keraSkipDec = SkipMissing('Keras')

@keraSkipDec
@noLogEntryExpected
def test_Keras_version():
    import keras
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

    layer0 = nimble.Init('Dense', units=64, activation='relu', input_dim=20)
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=1, activation='sigmoid')
    layers = [layer0, layer1, layer2]

    optimizer = nimble.Init('sgd', lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #####test fit
    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer=optimizer,
                       layers=layers, loss='binary_crossentropy', metrics=['accuracy'],
                       epochs=20, batch_size=128)

    #######test apply
    x = mym.apply(testX=x_train)

    #########test trainAndApply
    x = nimble.trainAndApply('keras.Sequential', trainX=x_train, trainY=y_train, testX=x_train,
                             optimizer=optimizer, layers=layers, loss='binary_crossentropy',
                             metrics=['accuracy'], epochs=20, batch_size=128)

    #########test trainAndTest
    x = nimble.trainAndTest('keras.Sequential', trainX=x_train, testX=x_train, trainY=y_train,
                            testY=y_train, optimizer=optimizer, layers=layers,
                            loss='binary_crossentropy', metrics=['accuracy'], epochs=20,
                            batch_size=128,
                            performanceFunction=nimble.calculate.loss.rootMeanSquareError)

    #########test CV
    results = nimble.crossValidate(
        "keras.Sequential", X=x_train, Y=y_train, optimizer=optimizer, layers=layers,
        loss='binary_crossentropy', metrics=['accuracy'], epochs=20, batch_size=128,
        performanceFunction=nimble.calculate.loss.rootMeanSquareError)
    bestArguments = results.bestArguments
    bestScore = results.bestResult

    #####test fit with Sequential object
    from keras.models import Sequential
    mym = nimble.train(Sequential, trainX=x_train, trainY=y_train, optimizer=optimizer,
                       layers=layers, loss='binary_crossentropy', metrics=['accuracy'],
                       epochs=20, batch_size=128, useLog=False)

@keraSkipDec
@logCountAssertionFactory(3)
def testKerasIncremental():
    """
    Test Keras can handle and incrementalTrain call
    """
    x_train = createData('Matrix', np.random.random((1000, 20)), useLog=False)
    y_train = createData('Matrix', np.random.randint(2, size=(1000, 1)), useLog=False)

    layer0 = nimble.Init('Dense', units=64, activation='relu', input_dim=20)
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=1, activation='sigmoid')
    layers = [layer0, layer1, layer2]
    optimizer = nimble.Init('sgd', lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #####test fit
    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer=optimizer,
                       layers=layers, loss='binary_crossentropy', metrics=['accuracy'],
                       epochs=20, batch_size=128)

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

    layer0 = nimble.Init('Dense', units=64, activation='relu', input_dim=7)
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=1, activation='sigmoid')
    layers = [layer0, layer1, layer2]
    optimizer = nimble.Init('sgd', lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer=optimizer,
                       layers=layers, loss='binary_crossentropy', metrics=['accuracy'],
                       epochs=2, steps_per_epoch=20, max_queue_size=1, steps=20)

    x = mym.apply(testX=x_train)
    assert len(x.points) == len(x_train.points)

@keraSkipDec
@logCountAssertionFactory(3)
def testKeras_TrainedLearnerApplyArguments():
    """ Test a keras function that accept arguments for predict"""
    x_train = createData('Matrix', np.random.random((1000, 20)), useLog=False)
    y_train = createData('Matrix', np.random.randint(2, size=(1000, 1)), useLog=False)

    layer0 = nimble.Init('Dense', units=64, activation='relu', input_dim=20)
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=1, activation='sigmoid')
    layers = [layer0, layer1, layer2]

    optimizer = nimble.Init('sgd', lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # Sequential.predict takes a 'steps' argument. Default is 20
    mym = nimble.train(
        'keras.Sequential', trainX=x_train, trainY=y_train,
        optimizer=optimizer, layers=layers, loss='binary_crossentropy',
        metrics=['accuracy'], epochs=1, shuffle=False)
    # using arguments parameter
    newArgs = mym.apply(testX=x_train, arguments={'steps': 50})
    # using kwarguments
    newArgs = mym.apply(testX=x_train, steps=50)

@keraSkipDec
@logCountAssertionFactory(1)
def testKeras_TrainedLearnerApplyArguments_exception():
    """ Test an keras function with invalid arguments for predict"""
    x_train = createData('Matrix', np.random.random((1000, 20)), useLog=False)
    y_train = createData('Matrix', np.random.randint(2, size=(1000, 1)), useLog=False)

    layer0 = nimble.Init('Dense', units=64, activation='relu', input_dim=20)
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=1, activation='sigmoid')
    layers = [layer0, layer1, layer2]

    optimizer = nimble.Init('sgd', lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # Sequential.predict does not take a 'foo' argument.
    mym = nimble.train(
        'keras.Sequential', trainX=x_train, trainY=y_train,
        optimizer=optimizer, layers=layers, loss='binary_crossentropy',
        metrics=['accuracy'], epochs=1, shuffle=False)
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


@raises(InvalidArgumentValue)
def testKeras_fitGeneratorOnlyParametersDisallowedForDense():
    x_data = np.random.random((20, 7))
    y_data = np.random.randint(2, size=(20, 1))

    x_train = createData('Matrix', x_data, useLog=False)
    y_train = createData('Matrix', y_data, useLog=False)

    layer0 = nimble.Init('Dense', units=64, activation='relu', input_dim=7)
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=1, activation='sigmoid')
    layers = [layer0, layer1, layer2]
    optimizer = nimble.Init('sgd', lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    # max_queue_size only applies to fit_generator for Sparse data
    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train,
                       optimizer=optimizer, layers=layers,
                       loss='binary_crossentropy', metrics=['accuracy'],
                       epochs=2, steps_per_epoch=20, max_queue_size=10)


@raises(InvalidArgumentValue)
def testKeras_fitOnlyParametersDisallowedForSparse():
    x_data = np.random.random((20, 7))
    y_data = np.random.randint(2, size=(20, 1))

    x_train = createData('Sparse', x_data, useLog=False)
    y_train = createData('Matrix', y_data, useLog=False)

    layer0 = nimble.Init('Dense', units=64, activation='relu', input_dim=7)
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=1, activation='sigmoid')
    layers = [layer0, layer1, layer2]
    optimizer = nimble.Init('sgd', lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    # max_queue_size only applies to fit_generator for Sparse data
    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train,
                       optimizer=optimizer, layers=layers,
                       loss='binary_crossentropy', metrics=['accuracy'],
                       epochs=2, steps_per_epoch=20, shuffle=True)
