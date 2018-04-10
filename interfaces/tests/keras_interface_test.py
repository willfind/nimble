"""
Unit tests for keras_interface.py

"""

from __future__ import absolute_import
import numpy.testing
from nose.plugins.attrib import attr

import UML
from UML import createData

from UML.interfaces.tests.test_helpers import checkLabelOrderingAndScoreAssociations

from UML.helpers import generateClusteredPoints

from UML.randomness import numpyRandom
from UML.exceptions import ArgumentException
from UML.helpers import generateClassificationData
from UML.helpers import generateRegressionData

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

def testKeras():
    """
    test keras fit, predict etc.
    """
    x_train = createData('Matrix', np.random.random((1000, 20)))
    y_train = createData('Matrix', np.random.randint(2, size=(1000, 1)))

    layers =[{'type':'Dense', 'units':64, 'activation':'relu', 'input_dim':20}, {'type':'Dropout', 'rate':0.5}, {'type':'Dense', 'units':1, 'activation':'sigmoid'}]
    #####test fit
    mym = UML.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer='sgd', layers=layers, loss=keras.losses.binary_crossentropy, metrics=['accuracy'], epochs=20, batch_size=128, lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    #######test apply
    x = mym.apply(testX=x_train)
    #########test trainAndApply
    x = UML.trainAndApply('keras.Sequential', trainX=x_train, trainY=y_train, testX=x_train, optimizer='sgd', \
                          layers=layers, loss=keras.losses.binary_crossentropy, metrics=['accuracy'], \
                          epochs=20, batch_size=128, lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #########test trainAndTest
    x = UML.trainAndTest('keras.Sequential', trainX=x_train, testX=x_train, trainY=y_train, testY=y_train, optimizer='sgd', layers=layers, loss=keras.losses.binary_crossentropy, metrics=['accuracy'], epochs=20, batch_size=128, lr=0.1, decay=1e-6, momentum=0.9, nesterov=True, performanceFunction=UML.calculate.loss.rootMeanSquareError)
    #########test CV
    bestArgument, bestScore = UML.crossValidateReturnBest("keras.Sequential", X=x_train, Y=y_train, optimizer='sgd', layers=layers, loss=keras.losses.binary_crossentropy, metrics=['accuracy'], epochs=20, batch_size=128, lr=0.1, decay=1e-6, momentum=0.9, nesterov=True, performanceFunction=UML.calculate.loss.rootMeanSquareError)

def testKerasIncremental():
    """
    test keras fit, predict etc.
    """
    x_train = createData('Matrix', np.random.random((1000, 20)))
    y_train = createData('Matrix', np.random.randint(2, size=(1000, 1)))

    layers =[{'type':'Dense', 'units':64, 'activation':'relu', 'input_dim':20}, {'type':'Dropout', 'rate':0.5}, {'type':'Dense', 'units':1, 'activation':'sigmoid'}]
    #####test fit
    mym = UML.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer='sgd', layers=layers, loss=keras.losses.binary_crossentropy, metrics=['accuracy'], epochs=20, batch_size=128, lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #####test incrementalTrain
    mym.incrementalTrain(x_train, y_train)

    x = mym.apply(testX=x_train)

def testKerasFitGenerator():
    """
    test keras fit_generator
    """
    x_data = np.random.random((20, 7))
    y_data = np.random.randint(2, size=(20, 1))

    x_train = createData('Sparse', x_data)
    y_train = createData('Matrix', y_data)

    layers =[{'type':'Dense', 'units':64, 'activation':'relu', 'input_dim':7}, {'type':'Dropout', 'rate':0.5}, {'type':'Dense', 'units':1, 'activation':'sigmoid'}]
    mym = UML.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer='sgd', layers=layers, loss=keras.losses.binary_crossentropy, metrics=['accuracy'], epochs=2, batch_size=1, lr=0.1, decay=1e-6, momentum=0.9, nesterov=True, steps_per_epoch=20, max_queue_size=1, shuffle=False)
    x = mym.apply(testX=x_train, steps=20)
