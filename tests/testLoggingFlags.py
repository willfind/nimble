# """
# Group of tests which checks that use controlled local and global
# mechanisms for controlling logging are functioning as expected.
#
# """
#
# from __future__ import absolute_import
# import os
#
# from nose.plugins.attrib import attr
#
# import UML
#
# from UML.helpers import generateClassificationData
# from UML.calculate import fractionIncorrect
# from UML.configuration import configSafetyWrapper
#
# learnerName = 'custom.KNNClassifier'
#
#
# # helper function which checks log staus
# def runAndCheck(toCall, useLog):
#     # generate data
#     cData = generateClassificationData(2, 10, 2)
#     ((trainX, trainY), (testX, testY)) = cData
#
#     # log file path
#     loc = UML.settings.get('logger', 'location')
#     name = UML.settings.get('logger', 'name')
#     # could check human readable or machine readable. we choose HR only,
#     # thus the addition of .txt
#     path = os.path.join(loc, name + '.txt')
#
#     if os.path.exists(path):
#         startSize = os.path.getsize(path)
#     else:
#         startSize = 0
#
#     # run given function
#     toCall(trainX, trainY, testX, testY, useLog)
#
#     # make sure it has the expected effect on the size
#     if os.path.exists(path):
#         endSize = os.path.getsize(path)
#     else:
#         endSize = 0
#
#     return (startSize, endSize)
#
#
# @configSafetyWrapper
# def backend(toCall):
#     # for each combination of local and global, call and check
#
#     UML.settings.set('logger', 'enabledByDefault', 'True')
#
#     (start, end) = runAndCheck(toCall, useLog=True)
#     assert start != end
#
#     (start, end) = runAndCheck(toCall, useLog=None)
#     assert start != end
#
#     (start, end) = runAndCheck(toCall, useLog=False)
#     assert start == end
#
#     UML.settings.set('logger', 'enabledByDefault', 'False')
#
#     (start, end) = runAndCheck(toCall, useLog=True)
#     assert start != end
#
#     (start, end) = runAndCheck(toCall, useLog=None)
#     assert start == end
#
#     (start, end) = runAndCheck(toCall, useLog=False)
#     assert start == end
#
#
# def test_train():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.train(learnerName, trainX, trainY, useLog=useLog)
#
#     backend(wrapped)
#
#
# def test_trainAndApply():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.trainAndApply(learnerName, trainX, trainY, testX, useLog=useLog)
#
#     backend(wrapped)
#
# #def test_trainAndApplyOvO():
# #	def wrapped(trainX, trainY, testX, testY, useLog):
# #		return UML.helpers.trainAndApplyOneVsOne(learnerName, trainX, trainY, testX, useLog=useLog)
#
# #	backend(wrapped)
#
# #def test_trainAndApplyOvA():
# #	def wrapped(trainX, trainY, testX, testY, useLog):
# #		return UML.helpers.trainAndApplyOneVsAll(learnerName, trainX, trainY, testX, useLog=useLog)
#
# #	backend(wrapped)
#
# def test_trainAndTest():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.trainAndTest(learnerName, trainX, trainY, testX, testY, performanceFunction=fractionIncorrect,
#                                 useLog=useLog)
#
#     backend(wrapped)
#
#
# def test_trainAndTestOnTrainingData_trainError():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.trainAndTestOnTrainingData(learnerName, trainX, trainY, performanceFunction=fractionIncorrect,
#                                               crossValidationError=False, useLog=useLog)
#
#     backend(wrapped)
#
#
# def test_trainAndTestOnTrainingData_CVError():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.trainAndTestOnTrainingData(learnerName, trainX, trainY, performanceFunction=fractionIncorrect,
#                                               crossValidationError=True, useLog=useLog)
#
#     backend(wrapped)
#
# #def test_trainAndTestOvO():
# #	def wrapped(trainX, trainY, testX, testY, useLog):
# #		return UML.helpers.trainAndTestOneVsOne(learnerName, trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, useLog=useLog)
#
# #	backend(wrapped)
#
# #def test_trainAndTestOvA():
# #	def wrapped(trainX, trainY, testX, testY, useLog):
# #		return UML.helpers.trainAndTestOneVsAll(learnerName, trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, useLog=useLog)
#
# #	backend(wrapped)
#
# def test_TrainedLearer_apply():
#     cData = generateClassificationData(2, 10, 2)
#     ((trainX, trainY), (testX, testY)) = cData
#     # get a trained learner
#     tl = UML.train(learnerName, trainX, trainY, useLog=False)
#
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return tl.apply(testX, useLog=useLog)
#
#     backend(wrapped)
#
#
# def test_TrainedLearer_test():
#     cData = generateClassificationData(2, 10, 2)
#     ((trainX, trainY), (testX, testY)) = cData
#     # get a trained learner
#     tl = UML.train(learnerName, trainX, trainY, useLog=False)
#
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return tl.test(testX, testY, performanceFunction=fractionIncorrect, useLog=useLog)
#
#     backend(wrapped)
#
#
# def test_crossValidate():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.crossValidate(learnerName, trainX, trainY, performanceFunction=fractionIncorrect, useLog=useLog)
#
#     backend(wrapped)
#
#
# def test_crossValidateReturnAll():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.crossValidateReturnAll(learnerName, trainX, trainY, performanceFunction=fractionIncorrect,
#                                           useLog=useLog)
#
#     backend(wrapped)
#
#
# def test_crossValidateReturnBest():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.crossValidateReturnBest(learnerName, trainX, trainY, performanceFunction=fractionIncorrect,
#                                            useLog=useLog)
#
#     backend(wrapped)
#
#
# # ??????????????????
# # incremental train?
# # reTrain?
# # createData?
#
# @configSafetyWrapper
# def backendDeep(toCall, setter):
#     UML.settings.set('logger', 'enabledByDefault', 'True')
#     setter('True')
#
#     # the deep logging flag is continget on global and local
#     # control, so we confirm that in those instances where
#     # logging should be disable, it is still disabled
#     (startT1, endT1) = runAndCheck(toCall, useLog=True)
#     (startT2, endT2) = runAndCheck(toCall, useLog=None)
#     (startT3, endT3) = runAndCheck(toCall, useLog=False)
#     assert startT3 == endT3
#
#     setter('False')
#
#     (startF1, endF1) = runAndCheck(toCall, useLog=True)
#     (startF2, endF2) = runAndCheck(toCall, useLog=None)
#     (startF3, endF3) = runAndCheck(toCall, useLog=False)
#     assert startF3 == endF3
#
#     # next we compare the differences between the calls when
#     # the deep flag is different
#     assert (endT1 - startT1) > (endF1 - startF1)
#     assert (endT2 - startT2) > (endF2 - startF2)
#
#     UML.settings.set('logger', 'enabledByDefault', 'False')
#     setter('True')
#
#     # the deep logging flag is continget on global and local
#     # control, so we confirm that logging is called or
#     # not appropriately
#     (startT1, endT1) = runAndCheck(toCall, useLog=True)
#     (startT2, endT2) = runAndCheck(toCall, useLog=None)
#     assert startT2 == endT2
#     (startT3, endT3) = runAndCheck(toCall, useLog=False)
#     assert startT3 == endT3
#
#     setter('False')
#
#     (startF1, endF1) = runAndCheck(toCall, useLog=True)
#     (startF2, endF2) = runAndCheck(toCall, useLog=None)
#     assert startF2 == endF2
#     (startF3, endF3) = runAndCheck(toCall, useLog=False)
#     assert startF3 == endF3
#
#     # next we compare the differences between the calls when
#     # the deep flag is different
#     assert (endT1 - startT1) > (endF1 - startF1)
#
#
# def test_Deep_crossValidate():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.crossValidate(learnerName, trainX, trainY, performanceFunction=fractionIncorrect, useLog=useLog)
#
#     def setter(val):
#         UML.settings.set('logger', 'enableCrossValidationDeepLogging', val)
#
#     backendDeep(wrapped, setter)
#
#
# def test_Deep_crossValidateReturnAll():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.crossValidateReturnAll(learnerName, trainX, trainY, performanceFunction=fractionIncorrect,
#                                           useLog=useLog)
#
#     def setter(val):
#         UML.settings.set('logger', 'enableCrossValidationDeepLogging', val)
#
#     backendDeep(wrapped, setter)
#
#
# def test_Deep_crossValidateReturnBest():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.crossValidateReturnBest(learnerName, trainX, trainY, performanceFunction=fractionIncorrect,
#                                            useLog=useLog)
#
#     def setter(val):
#         UML.settings.set('logger', 'enableCrossValidationDeepLogging', val)
#
#     backendDeep(wrapped, setter)
#
#
# def test_Deep_train():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         k = (2, 3)  # since we are not calling CV directly, we need to trigger it
#         return UML.train(learnerName, trainX, trainY, performanceFunction=fractionIncorrect, useLog=useLog, k=k)
#
#     def setter(val):
#         UML.settings.set('logger', 'enableCrossValidationDeepLogging', val)
#
#     backendDeep(wrapped, setter)
#
#
# def test_Deep_trainAndApply():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         k = (2, 3)  # since we are not calling CV directly, we need to trigger it
#         return UML.trainAndApply(learnerName, trainX, trainY, testX, performanceFunction=fractionIncorrect,
#                                  useLog=useLog, k=k)
#
#     def setter(val):
#         UML.settings.set('logger', 'enableCrossValidationDeepLogging', val)
#
#     backendDeep(wrapped, setter)
#
#
# def test_Deep_trainAndTest():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         k = (2, 3)  # since we are not calling CV directly, we need to trigger it
#         return UML.trainAndTest(learnerName, trainX, trainY, testX, testY, performanceFunction=fractionIncorrect,
#                                 useLog=useLog, k=k)
#
#     def setter(val):
#         UML.settings.set('logger', 'enableCrossValidationDeepLogging', val)
#
#     backendDeep(wrapped, setter)
#
# #def test_Deep_trainAndTestOnTrainingData_CVError():
# # TODO
#
# def test_Deep_TrainAndApplyOvO():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.trainAndApply(learnerName, trainX, trainY, testX, multiClassStrategy='OneVsOne', useLog=useLog)
#
#     def setter(val):
#         UML.settings.set('logger', 'enableMultiClassStrategyDeepLogging', val)
#
#     backendDeep(wrapped, setter)
#
#
# def test_Deep_trainAndApplyOVA():
#     def wrapped(trainX, trainY, testX, testY, useLog):
#         return UML.trainAndApply(learnerName, trainX, trainY, testX, multiClassStrategy='OneVsAll', useLog=useLog)
#
#     def setter(val):
#         UML.settings.set('logger', 'enableMultiClassStrategyDeepLogging', val)
#
#     backendDeep(wrapped, setter)
#
# #def test_Deep_TrainAndTestOvO():
# #	def wrapped(trainX, trainY, testX, testY, useLog):
# #		return UML.trainAndTest(learnerName, trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, multiClassStrategy='OneVsOne', useLog=useLog)
#
# #	def setter(val):
# #		UML.settings.set('logger', 'enableMultiClassStrategyDeepLogging', val)
#
# #	backendDeep(wrapped, setter)
#
# #def test_Deep_trainAndTestOVA():
# #	def wrapped(trainX, trainY, testX, testY, useLog):
# #		return UML.trainAndTest(learnerName, trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, multiClassStrategy='OneVsAll', useLog=useLog)
#
# #	def setter(val):
# #		UML.settings.set('logger', 'enableMultiClassStrategyDeepLogging', val)
#
# #	backendDeep(wrapped, setter)
