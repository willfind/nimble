"""
Unit tests for keras_interface.py

Cannot test for expected result equality due to inability to control for
randomness in Keras.
"""
import functools
import warnings

import numpy as np
import pytest
import sys
import os
import PIL
import datetime

import nimble
from nimble.core.interfaces.keras_interface import Keras
from nimble.exceptions import InvalidArgumentValue
from nimble._utility import DeferredModuleImport
from tests.helpers import raises
from tests.helpers import logCountAssertionFactory, noLogEntryExpected
from tests.helpers import skipMissingPackage
from tests.helpers import generateClassificationData
from tests.helpers import generateRegressionData


keras = DeferredModuleImport("keras")
tfKeras = DeferredModuleImport('tensorflow.keras')
tf = DeferredModuleImport("tensorflow")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
except ImportError:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout


keraSkipDec = skipMissingPackage('Keras')

@pytest.fixture
def optimizer():
    """
    Need fixture for parameter but value chosen by chooseOptimizer decorator.
    """

def chooseOptimizer(func):
    """
    Optimizer objects and/or parameters vary by keras API being used.
    """
    @functools.wraps(func)
    def wrapped(optimizer):
        kwOld = {'lr': 0.1, 'decay': 1e-6, 'momentum': 0.9, 'nesterov': True}
        # keras 2.3+ decay deprecated and lr changed to learning_rate
        kwNew =  {'learning_rate': 0.1, 'momentum': 0.9, 'nesterov': True}
        try:
            optimizer = nimble.Init('SGD', **kwNew)
            return func(optimizer=optimizer)
        except InvalidArgumentValue as IAV:
            if "When trying to validate arguments for " in str(IAV):
                pass
            raise IAV
        # execute this case outside of the except clause; makes for
        # slightly clearer output if there is a subsequent exception
        optimizer = nimble.Init('SGD', **kwOld)
        return func(optimizer=optimizer)

    return wrapped


@keraSkipDec
@noLogEntryExpected
def test_Keras_version():
    interface = Keras()
    if tfKeras.nimbleAccessible():
        # tensorflow.keras is prioritized based on recommendation from keras
        version = tfKeras.__version__
    elif keras.nimbleAccessible():
        version = keras.__version__
    assert interface.version() == version

@keraSkipDec
@logCountAssertionFactory(4)
@chooseOptimizer
def testKerasAPIClassification(optimizer):
    """
    Test Keras can handle a variety of arguments passed to all major learning functions
    """
    numClasses = 3
    allData = generateClassificationData(numClasses, 100, 16, 5)
    ((x_train, y_train), (x_test, y_test)) = allData

    x_np = x_train.copy(to="numpyarray")
    y_np = y_train.copy(to="numpyarray")
    x_np_test = x_test.copy(to="numpyarray")
    y_np_test = y_test.copy(to="numpyarray")

    layer0 = nimble.Init('Dense', units=64, activation='relu')
    layer0Raw = Dense(units=64, activation='relu')
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer1Raw = Dropout(rate=0.5)
    layer2 = nimble.Init('Dense', units=numClasses, activation='softmax')
    layer2Raw = Dense(units=numClasses, activation='softmax')
    layers = [layer0, layer1, layer2]
    layersRaw = [layer0Raw, layer1Raw, layer2Raw]
    #####test fit
    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer=optimizer,
                       layers=layers, loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                       epochs=20, batch_size=50)
    assert mym.learnerType == 'classification'

    #######test apply
    ret_A = mym.apply(testX=x_test)

    #########test trainAndApply
    ret_TaA = nimble.trainAndApply('keras.Sequential', trainX=x_train, trainY=y_train, testX=x_test,
                             optimizer=optimizer, layers=layers, loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'], epochs=20, batch_size=50)

    # Comparison to raw in-keras calculation. Different versions require different access
    try:
        if tfKeras.nimbleAccessible():
            opt_raw = getattr(tfKeras.optimizers, optimizer.name)(**optimizer.kwargs)
    except AttributeError:
        if keras.nimbleAccessible():
            opt_raw = getattr(keras.optimizers, optimizer.name)(**optimizer.kwargs)

    raw = Sequential(layersRaw)
    raw.compile(optimizer=opt_raw, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    raw.fit(x=x_np, y=y_np, epochs=20, batch_size=50)

    ret_raw_scores = raw.predict(x=x_np_test)
    # convert scores to labels
    ret_raw = np.argmax(ret_raw_scores, axis=1)
    ret_raw_obj = nimble.data(ret_raw.reshape(len(y_test.points),1), useLog=False)

    assert ret_A == ret_raw_obj
    assert ret_TaA == ret_raw_obj
    assert ret_raw_obj == y_test

    #########test trainAndTest
    FC = nimble.calculate.fractionCorrect
    ret_TaT = nimble.trainAndTest('keras.Sequential', FC, trainX=x_train,
                            testX=x_test, trainY=y_train, testY=y_test,
                            optimizer=optimizer, layers=layers,
                            loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                            epochs=20, batch_size=50)

    ### Comparison of trainAndTest results with raw predict results scores and evaluate
    ret_raw_test = FC(y_test, ret_raw_obj)
    raw_eval = raw.evaluate(x_np_test, y_np_test)
    assert ret_TaT == ret_raw_test
    # raw_eval[1] to get accuracy
    assert ret_TaT == raw_eval[1]


@keraSkipDec
@logCountAssertionFactory(4)
@chooseOptimizer
def testKerasBinaryClassificationOutputUnpacking(optimizer):
    binary = ['BinaryCrossentropy', 'Hinge', 'SquaredHinge']

    numClasses = 2
    allData = generateClassificationData(numClasses, 100, 8, 5)
    ((x_train, y_train), (x_test, y_test)) = allData

    y_train_cat = y_train.copy()
    y_train_cat.replaceFeatureWithBinaryFeatures(0, useLog=False)

    # All the losses here are 0/1 label to single logit/probability
    layer0 = nimble.Init('Dense', units=4, activation='linear', kernel_initializer="ones")
    layerlogit = nimble.Init('Dense', units=1, activation='linear', kernel_initializer="zeros")
    layerprob = nimble.Init('Dense', units=1, activation='sigmoid', kernel_initializer="zeros")

    FC = nimble.calculate.fractionCorrect

    for lossFunc in binary:
        argSets = []
        args = {"loss":nimble.Init(lossFunc), "layers":[layer0, layerprob]}
        argSets.append(args)

        # Want to try this with both from_logits True and False
        if lossFunc == "BinaryCrossentropy":
            args = {"loss":nimble.Init(lossFunc, from_logits=True),
                    "layers":[layer0, layerlogit]}
            argSets.append(args)

        for currArgs in argSets:
            ret_TaTL = nimble.trainAndTest('keras.Sequential', FC, trainX=x_train,
                                    testX=x_test, trainY=y_train, testY=y_test,
                                    optimizer=optimizer, metrics=['accuracy'],
                                    **currArgs)

            assert ret_TaTL == 1

@keraSkipDec
@logCountAssertionFactory(5)
@chooseOptimizer
def testKerasMultiClassificationOutputUnpacking(optimizer):
    multi = ["CategoricalCrossentropy", 'SparseCategoricalCrossentropy',
             'CategoricalHinge']

    numClasses = 3
    allData = generateClassificationData(numClasses, 100, 9, 5)
    ((x_train, y_train), (x_test, y_test)) = allData

    y_train_cat = y_train.copy()
    y_train_cat.replaceFeatureWithBinaryFeatures(0, useLog=False)

    # Categorical: one hot to number of classes of probabilities/logits
    # Sparse Cat: label to number of classes of probabilities/logits
    # Cat Hinge: one hot to singleton score
    layer0 = nimble.Init('Dense', units=3, activation='linear', kernel_initializer="ones")
    layerlogit = nimble.Init('Dense', units=3, activation='linear', kernel_initializer="zeros")
    layerprob = nimble.Init('Dense', units=3, activation='softmax', kernel_initializer="zeros")

    FC = nimble.calculate.fractionCorrect

    for lossFunc in multi:
        argSets = []
        print("lossFunc")
        if lossFunc == "CategoricalCrossentropy":
            args = {"layers": [layer0, layerlogit], "loss":nimble.Init(lossFunc, from_logits=True)}
            argSets.append(args)
            args = {"layers": [layer0, layerprob], "loss":nimble.Init(lossFunc, from_logits=False)}
            argSets.append(args)
            selectY = y_train_cat
        if lossFunc == "SparseCategoricalCrossentropy":
            args = {"layers": [layer0, layerlogit], "loss":nimble.Init(lossFunc, from_logits=True)}
            argSets.append(args)
            args = {"layers": [layer0, layerprob], "loss":nimble.Init(lossFunc, from_logits=False)}
            argSets.append(args)
            selectY = y_train
        if lossFunc == "CategoricalHinge":
            args = {"layers": [layer0, layerprob], "loss":nimble.Init(lossFunc)}
            argSets.append(args)
            selectY = y_train_cat

        for currArgs in argSets:
            ret_TaTL = nimble.trainAndTest('keras.Sequential', FC, trainX=x_train,
                                    testX=x_test, trainY=selectY, testY=y_test,
                                    optimizer=optimizer, metrics=['accuracy'],
                                    **currArgs)

            assert ret_TaTL == 1

def testKerasLayerSafety():
    numClasses = 3
    allData = generateClassificationData(numClasses, 100, 9, 5)
    ((x_train, y_train), (x_test, y_test)) = allData

    layer0 = nimble.Init('Dense', units=3, activation='linear', kernel_initializer="ones")
    layerprob = nimble.Init('Dense', units=3, activation='softmax', kernel_initializer="zeros")
    layers = [layer0, layerprob]

    FC = nimble.calculate.fractionCorrect

    _ = nimble.trainAndTest('keras.Sequential', FC, trainX=x_train,
                                    testX=x_test, trainY=y_train, testY=y_test,
                                    optimizer="adam", metrics=['accuracy'],
                                    layers=layers, loss="sparse_categorical_crossentropy")

    # layers list should still contain Init objects, not initialized layers
    for layer in layers:
        assert type(layer) == nimble.Init


@logCountAssertionFactory(4)
def testKerasAPIRegression():
    """
    Test Keras can handle a variety of arguments passed to all major learning functions
    """
    # Given the noisiness of the data, these operations are more suceptible to randomness,
    # so the random state is set to for each training cycle.
    seed = 144
    numClasses = 3
    allData = generateRegressionData(numClasses, 100, 8, 5)
    ((x_train, y_train), (x_test, y_test)) = allData

    x_np = x_train.copy(to="numpyarray")
    y_np = y_train.copy(to="numpyarray")
    x_np_test = x_test.copy(to="numpyarray")
    y_np_test = y_test.copy(to="numpyarray")

    layer0 = nimble.Init('Dense', units=16, kernel_initializer="zeros")
    layer0Raw = Dense(units=16, kernel_initializer="zeros")
    layer1 = nimble.Init('Dense', units=1, kernel_initializer="zeros")
    layer1Raw = Dense(units=1, kernel_initializer="zeros")
    layers = [layer0, layer1]
    layersRaw = [layer0Raw, layer1Raw]

    args = {"optimizer":"adam", "loss":"mean_absolute_error", "metrics":["mean_squared_error"],
            "epochs":50, "randomSeed":seed}
    #####test fit
    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, layers=layers, **args)
    assert mym.learnerType == 'regression'

    #######test apply
    ret_A = mym.apply(testX=x_test)

    #########test trainAndApply
    ret_TaA = nimble.trainAndApply('keras.Sequential', trainX=x_train, trainY=y_train, testX=x_test,
                             layers=layers, **args)

    # Comparison to raw in-keras calculation
    if tf.nimbleAccessible():
        tf.random.set_seed(seed)

    raw = Sequential(layersRaw)
    raw.compile(optimizer="adam", loss=args["loss"], metrics=args["metrics"])

    raw.fit(x=x_np, y=y_np, epochs=args["epochs"])

    ret_raw = raw.predict(x=x_np_test)
    ret_raw_obj = nimble.data(ret_raw.reshape(len(y_test.points),1), useLog=False)

    # different methods reproducible, both in and out of nimble
    assert ret_A == ret_raw_obj
    assert ret_TaA == ret_raw_obj

    #########test trainAndTest
    MAE = nimble.calculate.meanAbsoluteError
    ret_TaT = nimble.trainAndTest('keras.Sequential', MAE, trainX=x_train,
                            testX=x_test, trainY=y_train, testY=y_test,
                            layers=layers,**args)

    ### Comparison of trainAndTest results with raw predict results scores and evaluate
    ret_raw_test = MAE(y_test, ret_raw_obj)
    raw_eval = raw.evaluate(x_np_test, y_np_test)
    assert ret_TaT == ret_raw_test
    # raw_eval[1] to get loss
    assert ret_TaT == pytest.approx(raw_eval[0], rel=1e-4)
    assert ret_TaT < 1.5


@keraSkipDec
@logCountAssertionFactory(8)
@chooseOptimizer
def testKerasIncremental(optimizer):
    """
    Test Keras can handle and incrementalTrain call
    """
    numClasses = 3
    small = generateClassificationData(numClasses, 1, 16, 0)
    ((x_small, y_small), (_, _)) = small

    ret = generateClassificationData(numClasses, 100, 16, 10)
    ((x_train, y_train), (x_test, y_test)) = ret
    FC = nimble.calculate.fractionCorrect

    layer0 = nimble.Init('Dense', units=64, activation='relu')
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=numClasses, activation='softmax')
    layers = [layer0, layer1, layer2]
    ##### Poor Fit using a tiny portion of the data, without all of the labels represented
    # delete data associated with label 1
    x_small.points.delete(1,useLog=False)
    y_small.points.delete(1,useLog=False)
    mym = nimble.train('keras.Sequential', trainX=x_small, trainY=y_small, optimizer=optimizer,
                       layers=layers, loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                       epochs=1)

    before = mym.test(FC, x_test, y_test)

    ##### Better fit after 5 more batches over all the data, with all possible labels
    for _ in range(5):
        mym.incrementalTrain(x_train, y_train)

    after  = mym.test(FC, x_test, y_test)
    assert after > before

@keraSkipDec
@logCountAssertionFactory(2)
@chooseOptimizer
def testKeras_Sparse_Input(optimizer):
    """
    Test Keras on Sparse data; formerly uses different training method - fit_generator, should
    now usable with normal pipeline.
    """
    numClasses = 3
    allData = generateClassificationData(numClasses, 20, 16, 5)
    ((x_train, y_train), (x_test, y_test)) = allData

    layer0 = nimble.Init('Dense', units=64, activation='relu')
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=numClasses, activation='softmax')
    layers = [layer0, layer1, layer2]

    mym = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer=optimizer,
                       layers=layers, loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                       epochs=5, steps_per_epoch=20)

    ret = mym.apply(testX=x_test)
    assert ret == y_test

@keraSkipDec
@logCountAssertionFactory(3)
@chooseOptimizer
def testKeras_TrainedLearnerApplyArguments(optimizer):
    """ Test a keras function that accept arguments for predict"""
    x_train = nimble.data(np.random.random((1000, 20)), useLog=False)
    y_train = nimble.data(np.random.randint(2, size=(1000, 1)),
                          convertToType=float, useLog=False)

    layer0 = nimble.Init('Dense', units=64, activation='relu')
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=1, activation='sigmoid')
    layers = [layer0, layer1, layer2]

    # Sequential.predict takes a 'steps' argument. Default is 20
    mym = nimble.train(
        'keras.Sequential', trainX=x_train, trainY=y_train,
        optimizer=optimizer, layers=layers, loss='binary_crossentropy',
        metrics=['accuracy'], epochs=1, shuffle=False)

    # Setup the callback we'll be passing to apply
    try:
        if tfKeras.nimbleAccessible():
            CBBase = getattr(tfKeras.callbacks, "Callback")
    except AttributeError:
        if keras.nimbleAccessible():
            CBBase = getattr(keras.callbacks, "Callback")
    class CustomCallback(CBBase):
        def __init__(self):
            self.numHits = 0

        def on_predict_batch_begin(self, batch, logs=None):
            self.numHits += 1

    # using arguments parameter
    first = CustomCallback()
    assert first.numHits == 0
    _ = mym.apply(testX=x_train, arguments={'steps': 10, "callbacks":[first]})
    assert first.numHits == 10
    # using kwarguments
    second = CustomCallback()
    assert second.numHits == 0
    _ = mym.apply(testX=x_train, steps=50, callbacks=[second])
    assert second.numHits == 50

@keraSkipDec
@logCountAssertionFactory(1)
@chooseOptimizer
def testKeras_TrainedLearnerApplyArguments_exception(optimizer):
    """ Test an keras function with invalid arguments for predict"""
    x_train = nimble.data(np.random.random((1000, 20)), useLog=False)
    y_train = nimble.data(np.random.randint(2, size=(1000, 1)),
                          convertToType=float, useLog=False)

    layer0 = nimble.Init('Dense', units=64, activation='relu')
    layer1 = nimble.Init('Dropout', rate=0.5)
    layer2 = nimble.Init('Dense', units=1, activation='sigmoid')
    layers = [layer0, layer1, layer2]

    # Sequential.predict does not take a 'foo' argument.
    mym = nimble.train(
        'keras.Sequential', trainX=x_train, trainY=y_train,
        optimizer=optimizer, layers=layers, loss='binary_crossentropy',
        metrics=['accuracy'], epochs=1, shuffle=False)
    with raises(InvalidArgumentValue):
        # using arguments parameter
        newArgs1 = mym.apply(testX=x_train, arguments={'foo': 50})
    with raises(InvalidArgumentValue):
        # using kwarguments
        newArgs2 = mym.apply(testX=x_train, foo=50)

@keraSkipDec
@noLogEntryExpected
@chooseOptimizer
def testKerasReproducibility(optimizer):
    tfVersion1 = False
    try:
        from tensorflow import __version__
        if __version__[:2] == '1.':
            tfVersion1 = True
    except ImportError:
        pass

    if tfVersion1:
        # can't control randomness in this version. For testing, keras has
        # already been loaded, but by instantiating a new Keras instance we can
        # check that the warning will be displayed when users first use keras
        with warnings.catch_warnings(record=True) as rec:
            warnings.filterwarnings('always', module=r'.*keras_interface')
            _ = nimble.core.interfaces.keras_interface.Keras()
            start = "Randomness is outside of Nimble's control"
            assert rec and str(rec[0].message).startswith(start)
    else:
        # for version2 we expect reproducibility
        numClasses = 3
        x_data = nimble.random.numpyRandom.random((1000, 20))
        y_data = nimble.random.numpyRandom.randint(numClasses, size=(1000, 1))
        x_train = nimble.data(x_data, useLog=False)
        y_train = nimble.data(y_data, convertToType=float, useLog=False)

        nimble.random.setSeed(1234, useLog=False)
        layer0 = nimble.Init('Dense', units=64, activation='relu')
        layer1 = nimble.Init('Dropout', rate=0.5)
        layer2 = nimble.Init('Dense', units=numClasses, activation='sigmoid')
        layers = [layer0, layer1, layer2]
        mym1 = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer=optimizer,
                           layers=layers, loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                           epochs=20, batch_size=128, useLog=False)

        applied1 = mym1.apply(testX=x_train, useLog=False)


        nimble.random.setSeed(1234, useLog=False)
        layer0 = nimble.Init('Dense', units=64, activation='relu')
        layer1 = nimble.Init('Dropout', rate=0.5)
        layer2 = nimble.Init('Dense', units=numClasses, activation='sigmoid')
        layers = [layer0, layer1, layer2]
        mym2 = nimble.train('keras.Sequential', trainX=x_train, trainY=y_train, optimizer=optimizer,
                           layers=layers, loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                           epochs=20, batch_size=128, useLog=False)

        applied2 = mym2.apply(testX=x_train, useLog=False)

        assert applied1 == applied2

@keraSkipDec
def testLearnerTypes():
    learners = ['keras.' + l for l in nimble.learnerNames('keras')]
    assert all(lt == 'undefined' for lt in nimble.learnerType(learners))

@keraSkipDec
def testLearnerParameters():
    for learner in nimble.learnerNames('Keras'):
        params = nimble.learnerParameters('Keras.' + learner)
        # Params guaranteed by model to be in the compile API
        compileParams = ["optimizer", "loss", "metrics", "loss_weights",
                         "weighted_metrics"]
        # Common params shared by the keras apps loader functions
        appLoadParams = ["weights", "include_top", "input_shape",
                         "input_tensor", "classes"]
        if learner == "Sequential":
            checkIn = compileParams
            checkOut = appLoadParams
        else:
            checkIn = appLoadParams + compileParams
            checkOut = []

        for check in checkIn:
            assert check in params
        for check in checkOut:
            assert check not in params


@keraSkipDec
def testLossCoverage():
    kerasInt = nimble.core._learnHelpers.findBestInterface('keras')
    lts = nimble.core.interfaces.keras_interface.LEARNERTYPES
    fullLossList = lts["classification"] + lts["regression"]

    # For all Loss classes (which should cover all possiblilities that Keras
    # provides), check to see that it is represented in the LEARNERTYPES
    # constant.
    for n in dir(kerasInt.package.losses):
        val = getattr(kerasInt.package.losses, n)
        if type(val) is type and issubclass(val, kerasInt.package.losses.Loss):
            if not (val is kerasInt.package.losses.Loss):
                assert val.__name__ in fullLossList


def testLoadTrainedLearnerPredict():
    possible = nimble.learnerNames("keras")

    testImagesFolder = os.path.join(os.getcwd(), 'tests', 'interfaces', "testImages")

    allImages = []
    allImageIDs = []
    for imgName in sorted(os.listdir(testImagesFolder)):
        currPath = os.path.join(testImagesFolder, imgName)
        currIm = PIL.Image.open(currPath)
        currArr = np.array(currIm)
        currObj = nimble.data(currArr)
        dims = currObj.dimensions
        desired = [1] + list(dims)
        currObj.flatten()
        currObj.unflatten(desired)

        allImages.append(currObj)
        allImageIDs.append(imgName.split("_")[0])

    testY = nimble.data(allImageIDs, rowsArePoints=False)

    for name in possible:
        if name == "Sequential":
            continue
        # These aren't playing nice with the input_tensor trick
        if "VGG" in name:
            continue

        loadName = "keras." + name

        temp = nimble.Init("Input", shape=(None, None, 3))
        TL = nimble.loadTrainedLearner(loadName, weights="imagenet", input_tensor=temp)

        results = []
        for imgObj in allImages:
            ret = TL.apply(imgObj)[0]
            results.append(ret)

        retY = nimble.data(results, rowsArePoints=False)

        correct = nimble.calculate.fractionCorrect(testY, retY)
        assert correct >= 0.8
