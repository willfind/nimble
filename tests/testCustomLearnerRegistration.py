# import configparser
# import tempfile
# import os
# from unittest import mock
# import numpy
# import copy
# from nose.tools import raises
#
# import nimble
# from nimble import CustomLearner
# from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
# from nimble.learners import RidgeRegression
# from .assertionHelpers import configSafetyWrapper
# from .assertionHelpers import noLogEntryExpected
#
#
# class LoveAtFirstSightClassifier(CustomLearner):
#     """ Always predicts the value of the first class it sees in the most recently trained data """
#     learnerType = 'classification'
#
#     def incrementalTrain(self, trainX, trainY):
#         if hasattr(self, 'scope'):
#             self.scope = numpy.union1d(self.scope, trainY.copy(to='numpyarray').flatten())
#         else:
#             self.scope = numpy.unique(trainY.copy(to='numpyarray'))
#         self.prediction = trainY[0, 0]
#
#     def apply(self, testX):
#         ret = []
#         for point in testX.points:
#             ret.append([self.prediction])
#         return nimble.createData("Matrix", ret)
#
#     def getScores(self, testX):
#         ret = []
#         for point in testX.points:
#             currScores = []
#             for value in self.scope:
#                 if value == self.prediction:
#                     currScores.append(1)
#                 else:
#                     currScores.append(0)
#             ret.append(currScores)
#         return nimble.createData("Matrix", ret)
#
#     @classmethod
#     def options(cls):
#         return ['option']
#
#
# class UncallableLearner(CustomLearner):
#     learnerType = 'classification'
#
#     def train(self, trainX, trainY, foo):
#         return None
#
#     def apply(self, testX):
#         return None
#
#     @classmethod
#     def options(cls):
#         return ['option']
#
#
# @raises(InvalidArgumentValue)
# @configSafetyWrapper
# def testCustomPackageNameCollision():
#     """ Test registerCustomLearner raises an exception when the given name collides with a real package """
#     nonCustom = None
#     for inter in nimble.interfaces.available.values():
#         if not isinstance(inter, nimble.interfaces.CustomLearnerInterface):
#             nonCustom = inter
#             break
#     if nonCustom is not None:
#         nimble.registerCustomLearner(nonCustom.getCanonicalName(), LoveAtFirstSightClassifier)
#     else:
#         raise InvalidArgumentValue("Can't test, just pass")
#
#
# @raises(InvalidArgumentValue)
# @configSafetyWrapper
# def testDeregisterBogusPackageName():
#     """ Test deregisterCustomLearner raises an exception when passed a bogus customPackageName """
#     nimble.deregisterCustomLearner("BozoPackage", 'ClownClassifier')
#
#
# @raises(InvalidArgumentType)
# @configSafetyWrapper
# def testDeregisterFromNonCustom():
#     """ Test deregisterCustomLearner raises an exception when trying to deregister from a non-custom interface """
#     nimble.deregisterCustomLearner("Mlpy", 'KNN')
#
#
# @raises(InvalidArgumentValue)
# @configSafetyWrapper
# def testDeregisterBogusLearnerName():
#     """ Test deregisterCustomLearner raises an exception when passed a bogus learnerName """
#     #save state before registration, to check against final state
#     saved = copy.copy(nimble.interfaces.available)
#     try:
#         nimble.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
#         nimble.deregisterCustomLearner("Foo", 'BogusNameLearner')
#         # should never get to here
#         assert False
#     finally:
#         nimble.deregisterCustomLearner("Foo", "LoveAtFirstSightClassifier")
#         assert saved == nimble.interfaces.available
#
#
# @configSafetyWrapper
# def testMultipleCustomPackages():
#     """ Test registerCustomLearner correctly instantiates multiple custom packages """
#
#     #save state before registration, to check against final state
#     saved = copy.copy(nimble.interfaces.available)
#
#     nimble.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
#     nimble.registerCustomLearner("Bar", RidgeRegression)
#     nimble.registerCustomLearner("Baz", UncallableLearner)
#
#     assert nimble.listLearners("Foo") == ["LoveAtFirstSightClassifier"]
#     assert nimble.listLearners("Bar") == ["RidgeRegression"]
#     assert nimble.listLearners("Baz") == ["UncallableLearner"]
#
#     assert nimble.learnerParameters("Foo.LoveAtFirstSightClassifier") == set()
#     assert nimble.learnerParameters("Bar.RidgeRegression") == {'lamb'}
#     assert nimble.learnerParameters("Baz.UncallableLearner") == {'foo'}
#
#     nimble.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
#     nimble.deregisterCustomLearner("Bar", 'RidgeRegression')
#     nimble.deregisterCustomLearner("Baz", 'UncallableLearner')
#
#     assert saved == nimble.interfaces.available
#
#
# @configSafetyWrapper
# def testMultipleLearnersSinglePackage():
#     """ Test multiple learners grouped in the same custom package """
#     #TODO test is not isolated from above.
#
#     #save state before registration, to check against final state
#     saved = copy.copy(nimble.interfaces.available)
#
#     nimble.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
#     nimble.registerCustomLearner("Foo", RidgeRegression)
#     nimble.registerCustomLearner("Foo", UncallableLearner)
#
#     learners = nimble.listLearners("Foo")
#     assert len(learners) == 3
#     assert "LoveAtFirstSightClassifier" in learners
#     assert "RidgeRegression" in learners
#     assert "UncallableLearner" in learners
#
#     assert nimble.learnerParameters("Foo.LoveAtFirstSightClassifier") == set()
#     assert nimble.learnerParameters("Foo.RidgeRegression") == {'lamb'}
#     assert nimble.learnerParameters("Foo.UncallableLearner") == {'foo'}
#
#     nimble.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
#
#     learners = nimble.listLearners("Foo")
#     assert len(learners) == 2
#     assert "RidgeRegression" in learners
#     assert "UncallableLearner" in learners
#
#     nimble.deregisterCustomLearner("Foo", 'RidgeRegression')
#     nimble.deregisterCustomLearner("Foo", 'UncallableLearner')
#
#     assert saved == nimble.interfaces.available
#
#
# @configSafetyWrapper
# def testEffectToSettings():
#     """ Test that (de)registering is correctly recorded by nimble.settings """
#     nimble.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
#     regSec = 'RegisteredLearners'
#     opName = 'Foo.LoveAtFirstSightClassifier'
#     opValue = 'tests.testCustomLearnerRegistration.'
#     opValue += 'LoveAtFirstSightClassifier'
#
#     # check that it is listed immediately
#     assert nimble.settings.get(regSec, opName) == opValue
#
#     # check that it is *not* recorded to file
#     tempSettings = nimble._configuration.loadSettings()
#     try:
#         assert tempSettings.get(regSec, opName) == opValue
#         assert False  # expected NoOptionError
#     except configparser.NoOptionError:
#         pass
#
#     nimble.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
#
#     # check that deregistration removed the entry in the
#     # RegisteredLearners section
#     try:
#         assert nimble.settings.get(regSec, opName) == opValue
#         assert False
#     except configparser.NoOptionError:
#         pass
#
# # test that registering a sample custom learner with option names
# # will affect nimble.settings but not the config file
# @configSafetyWrapper
# def testRegisterLearnerWithOptionNames():
#     """ Test the availability of a custom learner's options """
#     nimble.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
#     learnerName = 'LoveAtFirstSightClassifier'
#
#     assert nimble.settings.get('Foo', learnerName + '.option') == ""
#     with open(nimble.settings.path) as config:
#         for line in config.readlines():
#             assert 'Foo' not in line
#
#     nimble.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
#
#
# # test what happens when you deregister a custom learner that has
# # in memory changes to its options in nimble.settings
# @configSafetyWrapper
# def testDeregisterWithSettingsChanges():
#     """ Safety check: unsaved options discarded after deregistration """
#     nimble.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
#     nimble.registerCustomLearner("Foo", UncallableLearner)
#
#     loveName = 'LoveAtFirstSightClassifier'
#     uncallName = 'UncallableLearner'
#     nimble.settings.set('Foo', loveName + '.option', "hello")
#     nimble.settings.set('Foo', uncallName + '.option', "goodbye")
#     assert nimble.settings.get('Foo', loveName + '.option') == "hello"
#
#     nimble.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
#
#     # should no longer have options for that learner in foo
#     try:
#         nimble.settings.get('Foo', loveName + '.option')
#         assert False
#     except configparser.NoOptionError:
#         pass
#
#     nimble.deregisterCustomLearner("Foo", 'UncallableLearner')
#
#     # should no longer have a section for foo
#     try:
#         nimble.settings.get('Foo', loveName + '.option')
#         assert False
#     except configparser.NoSectionError:
#         pass
#
#
# @configSafetyWrapper
# def test_settings_autoRegister():
#     """ Test that the auto registration helper correctly registers learners """
#     # establish that this custom package does not currently exist
#     try:
#         fooInt = nimble.helpers.findBestInterface('Foo')
#         assert False
#     except InvalidArgumentValue:
#         pass
#
#     # make changes to nimble.settings
#     name = 'Foo.LoveAtFirstSightClassifier'
#     value = LoveAtFirstSightClassifier.__module__ + '.' + 'LoveAtFirstSightClassifier'
#     nimble.settings.set("RegisteredLearners", name, value)
#
#     # call helper
#     nimble._configuration.autoRegisterFromSettings()
#
#     # check that correct learners are available through nimble.interfaces.available
#     assert 'LoveAtFirstSightClassifier' in nimble.listLearners('Foo')
#
#
# @configSafetyWrapper
# def test_settings_autoRegister_safety():
#     """ Test that the auto registration will not crash on strange inputs """
#
#     name = "Foo.Bar"
#     value = "Bogus.Attribute"
#     nimble.settings.set("RegisteredLearners", name, value)
#
#     # should throw warning, not exception
#     nimble._configuration.autoRegisterFromSettings()
#
# @configSafetyWrapper
# @noLogEntryExpected
# def test_logCount():
#     saved = copy.copy(nimble.interfaces.available)
#
#     nimble.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
#     lst = nimble.listLearners("Foo")
#     params = nimble.learnerParameters("Foo.LoveAtFirstSightClassifier")
#     defaults = nimble.learnerDefaultValues("Foo.LoveAtFirstSightClassifier")
#     lType = nimble.learnerType("Foo.LoveAtFirstSightClassifier")
#     nimble.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
#
#     nimble.registerCustomLearnerAsDefault("Bar", UncallableLearner)
#     lst = nimble.listLearners("Bar")
#     params = nimble.learnerParameters("Bar.UncallableLearner")
#     defaults = nimble.learnerDefaultValues("Bar.UncallableLearner")
#     lType = nimble.learnerType("Bar.UncallableLearner")
#     nimble.deregisterCustomLearnerAsDefault("Bar", 'UncallableLearner')
#
#     assert saved == nimble.interfaces.available
#
#
# # case 1: register new learner NOT as default
# @configSafetyWrapper
# def test_registerCustomLearnerNotWrittenToConfig():
#     saved = copy.copy(nimble.interfaces.available)
#     tmpConfig = tempfile.NamedTemporaryFile('w', suffix='.ini', delete=False)
#     # copy current config to tmpConfig
#     with open(nimble.settings.path, 'r') as config:
#         origLines = config.readlines()
#         for line in origLines:
#             tmpConfig.write(line)
#         tmpConfig.close()
#     origConfigSize = os.path.getsize(nimble.settings.path)
#     try:
#         newSession = nimble._configuration.SessionConfiguration(tmpConfig.name)
#         with mock.patch('nimble.settings', new=newSession):
#             with mock.patch('nimble.interfaces.available', new={}):
#                 nimble.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
#                 nimble.registerCustomLearner("Bar", RidgeRegression)
#                 nimble.registerCustomLearner("Baz", UncallableLearner)
#
#                 with open(tmpConfig.name, 'r') as testConfig:
#                     registeredLines = testConfig.readlines()
#
#                 # new learners are present in changes, but not in config
#                 regLearners = ['Foo.LoveAtFirstSightClassifier', 'Bar.RidgeRegression',
#                                'Baz.UncallableLearner']
#                 for learner in regLearners:
#                     assert learner in nimble.settings.changes['RegisteredLearners']
#
#                 # these two also have configurable options that should be
#                 # reflected in the settings
#                 opt1 = "LoveAtFirstSightClassifier.option"
#                 opt2 = "UncallableLearner.option"
#                 assert nimble.settings.get("Foo", opt1) == ""
#                 assert nimble.settings.get("Baz", opt2) == ""
#
#                 assert os.path.getsize(tmpConfig.name) == origConfigSize
#                 assert registeredLines == origLines
#
#                 nimble.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
#                 nimble.deregisterCustomLearner("Bar", 'RidgeRegression')
#                 nimble.deregisterCustomLearner("Baz", 'UncallableLearner')
#
#                 with open(tmpConfig.name, 'r') as testConfig:
#                     deregisteredLines = testConfig.readlines()
#
#                 # learners ToDelete in changes and removed from config
#                 for learner in regLearners:
#                     learnerObj = nimble.settings.changes['RegisteredLearners'][learner]
#                     assert isinstance(learnerObj, nimble._configuration.ToDelete)
#                 assert os.path.getsize(tmpConfig.name) == origConfigSize
#                 assert deregisteredLines == origLines
#     finally:
#         os.unlink(tmpConfig.name)
#         assert saved == nimble.interfaces.available
#
#
# # case 2: register new learner as default
# @configSafetyWrapper
# def test_registerCustomLearnerAsDefaultWrittenToConfig():
#     saved = copy.copy(nimble.interfaces.available)
#     tmpConfig = tempfile.NamedTemporaryFile('w', suffix='.ini', delete=False)
#     with open(nimble.settings.path, 'r') as config:
#         origLines = config.readlines()
#         for line in origLines:
#             tmpConfig.write(line)
#         tmpConfig.close()
#     origConfigSize = os.path.getsize(nimble.settings.path)
#     try:
#         newSession = nimble._configuration.SessionConfiguration(tmpConfig.name)
#         with mock.patch('nimble.settings', new=newSession):
#             with mock.patch('nimble.interfaces.available', new={}):
#                 nimble.registerCustomLearnerAsDefault("Foo", LoveAtFirstSightClassifier)
#                 nimble.registerCustomLearnerAsDefault("Bar", RidgeRegression)
#                 nimble.registerCustomLearnerAsDefault("Baz", UncallableLearner)
#
#                 with open(tmpConfig.name, 'r') as testConfig:
#                     registeredLines = testConfig.readlines()
#
#                 testLoc = 'tests.testCustomLearnerRegistration'
#                 fooLine = 'Foo.LoveAtFirstSightClassifier = '
#                 fooLine += testLoc + '.LoveAtFirstSightClassifier\n'
#                 barLine = 'Bar.RidgeRegression = '
#                 barLine += 'nimble.learners.ridge_regression.RidgeRegression\n'
#                 bazLine = 'Baz.UncallableLearner = '
#                 bazLine += testLoc + '.UncallableLearner\n'
#
#                 # changes should be empty; all learners should be written to config
#                 assert nimble.settings.changes == {}
#                 assert registeredLines != origLines
#                 for line in [fooLine, barLine, bazLine]:
#                     assert line in registeredLines and line not in origLines
#
#                 nimble.deregisterCustomLearnerAsDefault("Foo", 'LoveAtFirstSightClassifier')
#                 nimble.deregisterCustomLearnerAsDefault("Bar", 'RidgeRegression')
#                 nimble.deregisterCustomLearnerAsDefault("Baz", 'UncallableLearner')
#
#                 with open(tmpConfig.name, 'r') as testConfig:
#                     deregisteredLines = testConfig.readlines()
#
#                 # changes should be empty; all learners removed from config
#                 assert nimble.settings.changes == {}
#                 assert deregisteredLines == origLines
#                 for line in [fooLine, barLine, bazLine]:
#                     assert line not in deregisteredLines and line not in origLines
#     finally:
#         os.unlink(tmpConfig.name)
#         assert saved == nimble.interfaces.available
#
#
# # case 3: register new learner as default and deregister NOT as default
# @configSafetyWrapper
# def test_registerCustomLearnerAsDefaultDeregisterNotDefault():
#     saved = copy.copy(nimble.interfaces.available)
#     tmpConfig = tempfile.NamedTemporaryFile('w', suffix='.ini', delete=False)
#     with open(nimble.settings.path, 'r') as config:
#         origLines = config.readlines()
#         for line in origLines:
#             tmpConfig.write(line)
#         tmpConfig.close()
#     origConfigSize = os.path.getsize(nimble.settings.path)
#     try:
#         newSession = nimble._configuration.SessionConfiguration(tmpConfig.name)
#         with mock.patch('nimble.settings', new=newSession):
#             with mock.patch('nimble.interfaces.available', new={}):
#                 nimble.registerCustomLearnerAsDefault("Foo", LoveAtFirstSightClassifier)
#                 nimble.registerCustomLearnerAsDefault("Bar", RidgeRegression)
#                 nimble.registerCustomLearnerAsDefault("Baz", UncallableLearner)
#
#                 with open(tmpConfig.name, 'r') as testConfig:
#                     registeredLines = testConfig.readlines()
#
#                 testLoc = 'tests.testCustomLearnerRegistration'
#                 fooLine = 'Foo.LoveAtFirstSightClassifier = '
#                 fooLine += testLoc + '.LoveAtFirstSightClassifier\n'
#                 barLine = 'Bar.RidgeRegression = '
#                 barLine += 'nimble.learners.ridge_regression.RidgeRegression\n'
#                 bazLine = 'Baz.UncallableLearner = '
#                 bazLine += testLoc + '.UncallableLearner\n'
#
#                 # changes should be empty; all learners should be written to config
#                 assert nimble.settings.changes == {}
#                 assert registeredLines != origLines
#                 for line in [fooLine, barLine, bazLine]:
#                     assert line in registeredLines and line not in origLines
#
#                 nimble.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
#                 nimble.deregisterCustomLearner("Bar", 'RidgeRegression')
#                 nimble.deregisterCustomLearner("Baz", 'UncallableLearner')
#
#                 with open(tmpConfig.name, 'r') as testConfig:
#                     deregisteredLines = testConfig.readlines()
#
#                 # learners ToDelete in changes; all learners remain in config
#                 regLearners = ['Foo.LoveAtFirstSightClassifier', 'Bar.RidgeRegression',
#                                'Baz.UncallableLearner']
#                 for learner in regLearners:
#                     learnerObj = nimble.settings.changes['RegisteredLearners'][learner]
#                     assert isinstance(learnerObj, nimble._configuration.ToDelete)
#                 assert os.path.getsize(tmpConfig.name) > origConfigSize
#                 assert deregisteredLines == registeredLines
#
#                 # check that cannot access deregistered learners anymore
#                 # even though they are still present in the config file
#                 for interface in ['Foo', 'Bar', 'Baz']:
#                     try:
#                         nimble.listLearners(interface)
#                         assert False # expecting InvalidArgumentValue
#                     except InvalidArgumentValue:
#                         pass
#     finally:
#         os.unlink(tmpConfig.name)
#         assert saved == nimble.interfaces.available
#
#
# @configSafetyWrapper
# def test_registerCustomLearnerAsDefaultOnlyOneWrittenToConfig():
#     saved = copy.copy(nimble.interfaces.available)
#     tmpConfig = tempfile.NamedTemporaryFile('w', suffix='.ini', delete=False)
#     with open(nimble.settings.path, 'r') as config:
#         for line in config.readlines():
#             tmpConfig.write(line)
#         tmpConfig.close()
#     try:
#         newSession = nimble._configuration.SessionConfiguration(tmpConfig.name)
#         with mock.patch('nimble.settings', new=newSession):
#             with mock.patch('nimble.interfaces.available', new={}):
#                 nimble.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
#                 nimble.registerCustomLearner("Bar", RidgeRegression)
#                 nimble.registerCustomLearnerAsDefault("Baz", UncallableLearner)
#
#                 assert 'Foo.LoveAtFirstSightClassifier' in nimble.settings.changes['RegisteredLearners']
#                 assert 'Bar.RidgeRegression' in nimble.settings.changes['RegisteredLearners']
#                 assert 'Baz.UncallableLearner' not in nimble.settings.changes['RegisteredLearners']
#
#         with open(tmpConfig.name, 'r') as testConfig:
#             testLines = testConfig.readlines()
#         with open(nimble.settings.path, 'r') as origConfig:
#             origLines = origConfig.readlines()
#
#         assert testLines != origLines
#         testLoc = 'tests.testCustomLearnerRegistration'
#         fooLine = 'Foo.LoveAtFirstSightClassifier = '
#         fooLine += testLoc + '.LoveAtFirstSightClassifier\n'
#         assert fooLine not in testLines and fooLine not in origLines
#         barLine = 'Bar.RidgeRegression = '
#         barLine += 'nimble.learners.ridge_regression.RidgeRegression\n'
#         assert barLine not in testLines and barLine not in origLines
#         bazLine = 'Baz.UncallableLearner = '
#         bazLine += testLoc + '.UncallableLearner\n'
#         assert bazLine in testLines and bazLine not in origLines
#
#     finally:
#         os.unlink(tmpConfig.name)
#         assert saved == nimble.interfaces.available
