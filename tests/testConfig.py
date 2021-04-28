"""
Tests to check the loading, writing, and usage of nimble.settings, along
with the underlying structures being used.
"""

import tempfile
import copy
import os

from nose.tools import raises
from unittest import mock
import configparser

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction, PackageException


###############
### Helpers ###
###############

def getInterfaces(ignoreCustomInterfaces=False):
    interfaces = list(nimble.core.interfaces.available.values())
    if ignoreCustomInterfaces:
        interfaces = [int for int in interfaces
                      if int.getCanonicalName() not in ['nimble', 'custom']]
    return interfaces

class OptionNamedLookalike(object):
    def __init__(self, name, optNames):
        self.name = name
        self.optionNames = optNames

    def getCanonicalName(self):
        return self.name

    def isAlias(self, name):
        return name.lower() == self.getCanonicalName()


class FailedPredefined(object):
    def __init__(self):
        raise RuntimeError()

    @classmethod
    def getCanonicalName(cls):
        return 'FailedPredefined'

    @classmethod
    def isAlias(cls, name):
        return name.lower() == cls.getCanonicalName().lower()

    @classmethod
    def provideInitExceptionInfo(cls):
        raise PackageException("failed to load interface")

#############
### Tests ###
#############

def test_settings_GetSet():
    """ Test nimble.settings getters and setters """
    #orig changes
    origChangeSet = copy.deepcopy(nimble.settings.changes)

    # for available interfaces
    for interface in getInterfaces(ignoreCustomInterfaces=True):
        name = interface.getCanonicalName()
        if name in ['nimble', 'custom']:
            continue # custom learners do not have options
        for option in interface.optionNames:
            # get values of options
            origValue = nimble.settings.get(name, option)

            temp = "TEMPVALUE:" + name + option
            # change those values via nimble.settings -
            nimble.settings.set(name, option, temp)
            # check the change is reflected by all getters
            assert interface.getOption(option) == temp
            assert nimble.settings.get(name, option) == temp

            # change it back
            interface.setOption(option, origValue)
            # check again; taking into account default value substitution by
            # the interface
            if origValue != "":
                assert interface.getOption(option) == origValue
            else:
                assert interface.getOption(option) is None
            assert nimble.settings.get(name, option) == origValue

    # confirm that changes is the same
    assert nimble.settings.changes == origChangeSet


@raises(InvalidArgumentType)
def test_settings_HooksException_unCallable():
    """ Test SessionConfiguration.hook() throws exception on bad input """
    nimble.settings.hook("TestS", "TestOp", 5)


@raises(ImproperObjectAction)
def test_settings_HooksException_unHookable():
    """ Test SessionConfiguration.hook() throws exception for unhookable combo """
    nimble.settings.hook("TestS", "TestOp", None)

    def nothing(value):
        pass

    nimble.settings.hook("TestS", "TestOp", nothing)


@raises(InvalidArgumentValue)
def test_settings_HooksException_wrongSig():
    """ Test SessionConfiguration.hook() throws exception on incorrect signature """
    def twoArg(value, value2):
        pass

    nimble.settings.hook("TestS", "TestOp", twoArg)


def test_settings_Hooks():
    """ Test the on-change hooks for a SessionConfiguration object """
    history = []

    def appendToHistory(newValue):
        history.append(newValue)

    nimble.settings.hook("TestS", "TestOp", appendToHistory)

    nimble.settings.set("TestS", "TestOp", 5)
    nimble.settings.set("TestS", "TestOp", 4)
    nimble.settings.set("TestS", "TestOp", 1)
    nimble.settings.set("TestS", "TestOp", "Bang")

    assert history == [5, 4, 1, "Bang"]

def test_settings_GetFullConfig():
    """ Test nimble.settings.get when only specifying a section """
    startConfig = nimble.settings.get()
    assert 'logger' in startConfig
    loggerOptions = ['name', 'location', 'enabledByDefault',
                     'enableCrossValidationDeepLogging']
    assert all(opt in startConfig['logger'] for opt in loggerOptions)

    nimble.settings.setDefault("TestSec1", "op1", '1')
    nimble.settings.setDefault("TestSec1", "op2", '2')
    nimble.settings.set("TestSec1", "op1", '11')
    nimble.settings.set("TestSec2", "op1", '3')
    nimble.settings.set("TestSec2", "op2", '4')

    newConfig = nimble.settings.get()

    # everything in config from start should still be included
    for key, options in startConfig.items():
        for name, value in options.items():
            assert newConfig[key][name] == value

    assert "TestSec1" in newConfig
    assert 'TestSec2' in newConfig
    assert newConfig['TestSec1']["op1"] == '11'
    assert newConfig['TestSec1']["op2"] == '2'
    assert newConfig['TestSec2']["op1"] == '3'
    assert newConfig['TestSec2']["op2"] == '4'


def test_settings_GetSectionOnly():
    """ Test nimble.settings.get when only specifying a section """
    nimble.settings.set("TestSec1", "op1", '1')
    nimble.settings.set("TestSec1", "op2", '2')

    allSec1 = nimble.settings.get("TestSec1")
    assert allSec1["op1"] == '1'
    assert allSec1['op2'] == '2'


def test_settings_saving():
    """ Test nimble.settings will save its in memory changes """
    # make some change via nimble.settings. save it,
    nimble.settings.set("newSectionName", "new.Option.Name", '1')
    nimble.settings.saveChanges()

    # reload it with the starup function, make sure settings saved.
    nimble.settings = nimble.core.configuration.loadSettings()
    assert nimble.settings.changes == {}
    assert nimble.settings.get("newSectionName", 'new.Option.Name') == '1'


def test_settings_savingSection():
    """ Test nimble.settings.saveChanges when specifying a section """
    nimble.settings.changes = {}

    nimble.settings.set("TestSec1", "op1", '1')
    nimble.settings.set("TestSec1", "op2", '2')
    nimble.settings.set("TestSec2", "op1", '1')
    nimble.settings.saveChanges("TestSec1")

    # assert that other changes are still in effect
    assert len(nimble.settings.changes) == 1
    assert nimble.settings.get("TestSec2", "op1") == '1'

    # reload it with the starup function, make sure settings saved.
    temp = nimble.core.configuration.loadSettings()
    assert temp.get('TestSec1', "op1") == '1'
    assert temp.get('TestSec1', "op2") == '2'
    # confirm that the change outside the section was not saved
    try:
        val = temp.get('TestSec2', "op1")
        assert False
    except configparser.NoSectionError:
        pass


def test_settings_savingOption():
    """ Test nimble.settings.saveChanges when specifying a section and option """
    nimble.settings.changes = {}

    nimble.settings.set("TestSec1", "op1", '1')
    nimble.settings.set("TestSec1", "op2", '2')
    nimble.settings.set("TestSec2", "op1", '1')
    nimble.settings.saveChanges("TestSec1", "op2")

    # assert that other changes are still in effect
    assert len(nimble.settings.changes["TestSec1"]) == 1
    assert len(nimble.settings.changes["TestSec2"]) == 1
    assert len(nimble.settings.changes) == 2
    assert nimble.settings.get("TestSec2", "op1") == '1'
    assert nimble.settings.get("TestSec1", "op1") == '1'

    # reload it with the starup function, make that option was saved.
    temp = nimble.core.configuration.loadSettings()
    assert temp.get('TestSec1', "op2") == '2'
    # confirm that the other changes were not saved
    try:
        val = temp.get('TestSec2', "op1")
        assert False
    except configparser.NoSectionError:
        pass
    try:
        val = temp.get('TestSec1', "op1") == '1'
        assert False
    except configparser.NoOptionError:
        pass

def setAvailableInterfaceOptions(save=False):
    """
    Set and save the options for each available interface.
    """
    for interface in getInterfaces(ignoreCustomInterfaces=True):
        nimble.core.configuration.setInterfaceOptions(interface, save)

def test_settings_addingNewInterface():
    """ Test nimble.core.configuration.setInterfaceOptions correctly sets options """
    tempInterface = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
    nimble.core.interfaces.available[tempInterface.name] = tempInterface
    ignoreInterface = OptionNamedLookalike("ig", [])
    nimble.core.interfaces.available[ignoreInterface.name] = ignoreInterface

    # set options for all interfaces
    setAvailableInterfaceOptions()

    # make sure there is no section associated with the optionless
    # interface
    assert not nimble.settings.parser.has_section('ig')

    # make sure new section and name was correctly added
    # '' is default value when adding options from interfaces
    assert nimble.settings.get('Test', 'Temp0') == ''
    assert nimble.settings.get('Test', 'Temp1') == ''

    # reload settings - options should not be recorded
    nimble.settings = nimble.core.configuration.loadSettings()

    assert not nimble.settings.parser.has_section('ig')
    assert not nimble.settings.parser.has_section('Test')

    # save options for all interfaces and reload
    setAvailableInterfaceOptions(True)
    nimble.settings = nimble.core.configuration.loadSettings()

    # make sure there is no section associated with the optionless
    # interface
    assert not nimble.settings.parser.has_section('ig')

    # make sure new section and name was correctly added
    # '' is default value when adding options from interfaces
    assert nimble.settings.get('Test', 'Temp0') == ''
    assert nimble.settings.get('Test', 'Temp1') == ''

def test_settings_setInterfaceOptionsSafety():
    """ Test that setting options preserves values already in the config file """
    tempInterface1 = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
    nimble.core.interfaces.available[tempInterface1.name] = tempInterface1
    tempInterface2 = OptionNamedLookalike("TestOther", ['Temp0'])
    nimble.core.interfaces.available[tempInterface2.name] = tempInterface2

    # set options for all interfaces
    setAvailableInterfaceOptions()

    assert nimble.settings.get("Test", 'Temp0') == ''
    assert nimble.settings.get("Test", 'Temp1') == ''
    assert nimble.settings.get("TestOther", 'Temp0') == ''

    # save options for Test, but not for TestOther
    nimble.settings.set('Test', 'Temp0', '0')
    nimble.settings.set('Test', 'Temp1', '1')
    nimble.settings.saveChanges()

    nimble.settings.set('TestOther', 'Temp0', '0')

    assert nimble.settings.get("TestOther", 'Temp0') == '0'

    # reload - TestOther should revert to default ''
    nimble.settings = nimble.core.configuration.loadSettings()

    assert nimble.settings.get("Test", 'Temp0') == '0'
    assert nimble.settings.get("Test", 'Temp1') == '1'
    assert nimble.settings.get("TestOther", 'Temp0') == ''


def test_settings_setInterfaceOptionsChanges():
    """ Test that setting interface options properly saves current changes """
    tempInterface1 = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
    tempInterface2 = OptionNamedLookalike("TestOther", ['Temp0'])
    nimble.core.interfaces.available[tempInterface1.name] = tempInterface1
    nimble.core.interfaces.available[tempInterface2.name] = tempInterface2

    # set with new interfaces
    setAvailableInterfaceOptions()

    nimble.settings.set('Test', 'Temp0', '0')
    nimble.settings.set('Test', 'Temp1', '1')
    nimble.settings.set('TestOther', 'Temp0', 'unchanged')

    assert nimble.settings.get('Test', 'Temp0') == '0'

    # change Test option names and reset options for all interfaces
    tempInterface1.optionNames[1] = 'NotTemp1'
    setAvailableInterfaceOptions()

    # NOTE: our interfaces do not currently allow for option names to be
    # changed, but in that case we would want the following behavior

    # check value of unchanged option
    assert nimble.settings.get('Test', 'Temp0') == '0'
    # previous option and value are available and unchanged
    # we do not delete as this could become a valid option again later
    assert nimble.settings.get('Test', 'Temp1') == '1'
    # new option now available
    assert nimble.settings.get('Test', 'NotTemp1') == ''
    # check that the temp value for TestOther is unaffected
    assert nimble.settings.get('TestOther', 'Temp0') == 'unchanged'
    # set should only allow setting for new option
    nimble.settings.set('Test', 'NotTemp1', '2')
    try:
        nimble.settings.set('Test', 'Temp1', '2')
        assert False
    except InvalidArgumentValue:
        pass

    # now change the option back to the original
    tempInterface1.optionNames[1] = 'Temp1'
    setAvailableInterfaceOptions()

    nimble.settings.set('Test', 'Temp1', '2')
    try:
        nimble.settings.set('Test', 'NotTemp1', '2')
        assert False
    except InvalidArgumentValue:
        pass

def test_settings_allowedNames():
    """ Test that you can only set allowed names in interface sections """
    for interface in getInterfaces(ignoreCustomInterfaces=True):
        name = interface.getCanonicalName()
        try:
            nimble.settings.set(name, 'foo', "bar")
            assert False
        except InvalidArgumentValue:
            pass

def test_settings_customLearnerOptionsException():
    """ Test that you cannot set options for custom learner interfaces """

    for name in ['nimble', 'custom']:
        try:
            nimble.settings.set(name, 'foo', "bar")
            assert False # expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

@raises(configparser.NoSectionError)
# test that set without save is temporary
def test_settings_set_without_save():
    # make some change via nimble.settings.
    nimble.settings.set("tempSectionName", "temp.Option.Name", '1')
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name') == '1'

    # reload it with the startup function, try to load something which
    # shouldn't be there
    nimble.settings = nimble.core.configuration.loadSettings()
    nimble.settings.get("tempSectionName", 'temp.Option.Name')

def test_settings_setDefault():
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name2')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass

    nimble.settings.set("tempSectionName", "temp.Option.Name1", '1')
    nimble.settings.setDefault("tempSectionName", "temp.Option.Name2", '2')
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    # Name2 should be reflected in file, but not Name1
    nimble.settings = nimble.core.configuration.loadSettings()
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except configparser.NoOptionError:
        pass

    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'


@mock.patch('nimble.core.interfaces.predefined', [FailedPredefined])
def testSetLocationForFailedPredefinedInterface():
    nimble.settings.set('FailedPredefined', 'location', 'path/to/mock')


@raises(InvalidArgumentValue)
@mock.patch('nimble.core.interfaces.predefined', [FailedPredefined])
def testExceptionSetOptionForFailedPredefinedInterface():
    nimble.settings.set('FailedPredefined', 'foo', 'path/to/mock')
