
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

import pytest

import nimble
from nimble.calculate import fractionIncorrect
from nimble._utility import storm_tuner
from tests.helpers import logCountAssertionFactory
from tests.interfaces.keras_interface_test import keraSkipDec

# TODO other Selectors and Validators

noStormTuner = not storm_tuner.nimbleAccessible()

@pytest.mark.slow
@logCountAssertionFactory(2)
@keraSkipDec
@pytest.mark.skipif(noStormTuner, reason='storm_tuner is not available')
def test_Storm_learnerArgsFunc():

    def modelArguments(hp):
        layers = []
        kernel0 = hp.Param('kernelSize0', [64, 128, 256], ordered=True)
        activation = hp.Param('activation', ['relu', 'elu'])
        # layers created from the current choice for the above arguments
        layers.append(nimble.Init('Dense', units=kernel0))
        layers.append(nimble.Init('Activation', activation=activation))
        # access the current kernelSize0 value to determine future units
        kernelSize = hp.values['kernelSize0']
        # build a variable number of hidden layers
        for x in range(hp.Param('num_layers', [1, 2, 3], ordered=True)):
            kernelSize = int(0.75 * kernelSize)
            layers.append(nimble.Init('Dense', units=kernelSize))
            layers.append(nimble.Init('Activation',
                                      activation=activation))

        layers.append(nimble.Init('Dense', units=3,
                                  activation='sigmoid'))

        return {'layers': layers}

    trainX = nimble.random.data(100, 10, 0, useLog=False)
    trainY = nimble.random.data(100, 1, 0, elementType='int', useLog=False) // 34
    valX = nimble.random.data(10, 10, 0, useLog=False)
    valY = nimble.random.data(10, 1, 0, elementType='int', useLog=False) // 34
    # Without hyperparameter tuning, layers would need to be provided
    # but here modelArguments will determine the best layers.
    tl = nimble.train('Keras.Sequential', trainX, trainY, epochs=5,
                     tuning=nimble.Tuning('storm', 'data', fractionIncorrect,
                                          validateX=valX, validateY=valY,
                                          learnerArgsFunc=modelArguments,
                                          maxIterations=10),
                     loss='sparse_categorical_crossentropy',
                     optimizer=nimble.Tune(['Adam', 'Adamax']),
                     metrics=['accuracy'], verbose=0)

    assert 'layers' in tl.arguments # generated by learnerArgsFunc
    assert tl.arguments['optimizer'] in ['Adam', 'Adamax']
    nimble.showLog()
