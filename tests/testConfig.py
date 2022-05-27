"""
Tests to check the loading, writing, and usage of nimble.settings, along
with the underlying structures being used.
"""

import os
import copy
import pathlib
from functools import wraps
import tempfile
import sys
import io
import re
import importlib

import configparser

import nimble
from nimble.core.configuration import SessionConfiguration
from nimble.core.interfaces.universal_interface import PredefinedInterfaceMixin
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction, PackageException
from nimble._dependencies import DEPENDENCIES
from tests.helpers import raises, patch


###############
### Helpers ###
###############

def getInterfaces(ignoreCustomInterfaces=False):
    interfaces = list(nimble.core.interfaces.available.values())
    if ignoreCustomInterfaces:
        interfaces = [int for int in interfaces
                      if int.getCanonicalName() not in ['nimble', 'custom']]
    return interfaces

def optionNamedLookalikeFactory(name, optNames):
    class OptionNamedLookalike(object):
        @classmethod
        def optionNames(cls):
            return optNames

        @classmethod
        def getCanonicalName(cls):
            return name

        @classmethod
        def isAlias(cls, name):
            return name.lower() == cls.getCanonicalName()

    return OptionNamedLookalike()

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

def useSessionConfiguration(test):
    """
    Wrapper for tests that use saveChanges or setDefault.

    All tests override nimble.settings so that they can be run without using a
    file. This override redefines saveChanges, so here we want to reinstate the
    use of SessionConfiguration for nimble.settings so that we can test that
    changes are being saved to the configuration file as expected.
    """
    @wraps(test)
    def wrapped():
        with tempfile.NamedTemporaryFile('w+', suffix='.ini') as tmp:
            # can reuse the settings established when test began
            # will copy everything not in changes to the temporary config file
            # and store changes to apply to new nimble.settings object
            changes = nimble.settings.changes
            for section, options in nimble.settings.get().items():
                if section not in changes:
                    tmp.write('[{}]\n'.format(section))
                    for option in options.items():
                        tmp.write('{}\n'.format(' = '.join(option)))

            def loadSavedSettings():
                tmp.seek(0)
                return SessionConfiguration(tmp.name)

            nimble.settings = loadSavedSettings()
            nimble.settings.changes = copy.deepcopy(changes)
            # for tests that load settings during the test
            nimble.core.configuration.loadSettings = loadSavedSettings

            test()
            # nimble.settings is reset at the beginning of each test so there
            # is no need to restore anything after the test completes.

    return wrapped

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
        for option in interface.optionNames():
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

    try:
        nimble.settings.hook("TestS", "TestOp", nothing)
    finally:
        del nimble.settings.hooks[("TestS", "TestOp")]


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

@useSessionConfiguration
def test_settings_GetFullConfig():
    """ Test nimble.settings.get when only specifying a section """
    startConfig = nimble.settings.get()
    assert 'logger' in startConfig
    loggerOptions = ['name', 'location', 'enabledByDefault',
                     'enableDeepLogging']
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

@useSessionConfiguration
def test_settings_saving():
    """ Test nimble.settings will save its in memory changes """
    # make some change via nimble.settings. save it,
    nimble.settings.set("newSectionName", "new.Option.Name", '1')
    nimble.settings.saveChanges()

    # reload it with the starup function, make sure settings saved.
    nimble.settings = nimble.core.configuration.loadSettings()
    assert nimble.settings.changes == {}
    assert nimble.settings.get("newSectionName", 'new.Option.Name') == '1'

@useSessionConfiguration
def test_settings_savingSection():
    """ Test nimble.settings.saveChanges when specifying a section """
    nimble.settings.changes = {}

    # Ensure no bleed from other tests
    with raises(configparser.NoSectionError):
        _ = nimble.settings.get('TestSec2', "op1")

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
    with raises(configparser.NoSectionError):
        val = temp.get('TestSec2', "op1")

@useSessionConfiguration
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
    with raises(configparser.NoSectionError):
        val = temp.get('TestSec2', "op1")
    with raises(configparser.NoOptionError):
        val = temp.get('TestSec1', "op1") == '1'

def setAvailableInterfaceOptions(save=False):
    """
    Set and save the options for each available interface.
    """
    for interface in getInterfaces(ignoreCustomInterfaces=True):
        nimble.core.configuration.setInterfaceOptions(interface, save)

def test_settings_addingNewInterface():
    """ Test nimble.core.configuration.setInterfaceOptions correctly sets options """
    name = "Test"
    tempInterface = optionNamedLookalikeFactory(name, ['Temp0', 'Temp1'])
    nimble.core.interfaces.available[name] = tempInterface
    name = "ig"
    ignoreInterface = optionNamedLookalikeFactory(name, [])
    nimble.core.interfaces.available[name] = ignoreInterface

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

@useSessionConfiguration
def test_settings_setInterfaceOptionsSafety():
    """ Test that setting options preserves values already in the config file """
    name = "Test"
    tempInterface1 = optionNamedLookalikeFactory(name, ['Temp0', 'Temp1'])
    nimble.core.interfaces.available[name] = tempInterface1
    name = "TestOther"
    tempInterface2 = optionNamedLookalikeFactory(name, ['Temp0'])
    nimble.core.interfaces.available[name] = tempInterface2

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
    name1 = "Test"
    name2 = "TestOther"
    tempInterface1 = optionNamedLookalikeFactory(name1, ['Temp0', 'Temp1'])
    tempInterface2 = optionNamedLookalikeFactory(name2, ['Temp0'])
    nimble.core.interfaces.available[name1] = tempInterface1
    nimble.core.interfaces.available[name2] = tempInterface2

    # set with new interfaces
    setAvailableInterfaceOptions()

    nimble.settings.set('Test', 'Temp0', '0')
    nimble.settings.set('Test', 'Temp1', '1')
    nimble.settings.set('TestOther', 'Temp0', 'unchanged')

    assert nimble.settings.get('Test', 'Temp0') == '0'

    # change Test option names and reset options for all interfaces
    with patch(tempInterface1, "optionNames", lambda: ["Temp0", "NotTemp1"]):
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
        with raises(InvalidArgumentValue):
            nimble.settings.set('Test', 'Temp1', '2')

    # outside of patch, option name now back to the original
    setAvailableInterfaceOptions()

    nimble.settings.set('Test', 'Temp1', '2')
    with raises(InvalidArgumentValue):
        nimble.settings.set('Test', 'NotTemp1', '2')

def test_settings_allowedNames():
    """ Test that you can only set allowed names in interface sections """
    for interface in getInterfaces(ignoreCustomInterfaces=True):
        name = interface.getCanonicalName()
        with raises(InvalidArgumentValue):
            nimble.settings.set(name, 'foo', "bar")

def test_settings_customLearnerOptionsException():
    """ Test that you cannot set options for custom learner interfaces """

    for name in ['nimble', 'custom']:
        with raises(InvalidArgumentValue):
            nimble.settings.set(name, 'foo', "bar")

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

@useSessionConfiguration
def test_settings_setDefault():
    with raises(configparser.NoSectionError):
        nimble.settings.get("tempSectionName", 'temp.Option.Name2')

    nimble.settings.set("tempSectionName", "temp.Option.Name1", '1')
    nimble.settings.setDefault("tempSectionName", "temp.Option.Name2", '2')
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    # Name2 should be reflected in file, but not Name1
    nimble.settings = nimble.core.configuration.loadSettings()
    with raises(configparser.NoOptionError):
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')

    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

def test_settings_str():
    """ Test nimble.settings string has expected sections and options """

    s = str(nimble.settings)
    # check that string is valid for configparser and use configparser to test
    # that it contains the expected sections and options for logger and fetch
    cp = configparser.ConfigParser()
    cp.read_string(s)
    assert 'logger' in cp
    assert 'location' in cp['logger']
    assert 'enabledByDefault' in cp['logger']
    assert 'enableDeepLogging' in cp['logger']
    assert 'fetch' in cp
    assert 'location' in cp['fetch']

def test_settings_repr():
    """ Test nimble.settings __repr__ has expected sections and options """

    s = repr(nimble.settings)
    _, s = s.split('\n', 1)
    s, _ = s.rsplit('\n', 1)
    # check that string is valid for configparser and use configparser to test
    # that it contains the expected sections and options for logger and fetch
    cp = configparser.ConfigParser()
    cp.read_string(s)
    assert 'logger' in cp
    assert 'location' in cp['logger']
    assert 'enabledByDefault' in cp['logger']
    assert 'enableDeepLogging' in cp['logger']
    assert 'fetch' in cp
    assert 'location' in cp['fetch']

@patch(nimble.core.interfaces, 'predefined', [FailedPredefined])
def testSetLocationForFailedPredefinedInterface():
    nimble.settings.set('FailedPredefined', 'location', 'path/to/mock')


@raises(InvalidArgumentValue)
@patch(nimble.core.interfaces, 'predefined', [FailedPredefined])
def testExceptionSetOptionForFailedPredefinedInterface():
    nimble.settings.set('FailedPredefined', 'foo', 'path/to/mock')

# copied from nimble.core.configuration
def loadSettings():
    """
    Function which reads the configuration file and loads the values
    into a SessionConfiguration. The SessionConfiguration object is then
    returned.
    """
    configFile = '.nimble.ini'
    currPath = os.path.join(os.getcwd(), configFile)
    homeLoc = str(pathlib.Path.home())
    homePath = os.path.join(homeLoc, configFile)

    if not os.path.exists(currPath):
        if not os.path.exists(homePath):
            with open(homePath, 'w'):
                pass
        target = homePath
    else:
        target = currPath

    ret = SessionConfiguration(target)

    sections = ret.get()
    if 'fetch' not in sections or 'location' not in sections['fetch']:
        ret.setDefault('fetch', 'location', homeLoc)

    return ret

def test_settings_initUsesHomeOrCWDFile():
    cwd = os.getcwd()
    tempDir = nimble.settings.get('logger', 'location')
    try:
        os.chdir(tempDir)
        assert cwd != os.getcwd()
        currConfigPath = os.path.join(os.getcwd(), '.nimble.ini')
        assert not os.path.exists(currConfigPath)
        nimble.settings = loadSettings()
        home = str(pathlib.Path.home())
        assert nimble.settings.path == os.path.join(home, '.nimble.ini')
        with open('.nimble.ini', 'w'):
            pass
        nimble.settings = loadSettings()
        assert nimble.settings.path == currConfigPath
    finally:
        os.chdir(cwd)

def back_checkStatus(availability, expectInvalid=False):
    saved = sys.stdout
    try:
        stream = io.StringIO()
        sys.stdout = stream
        nimble.showAvailablePackages()
    finally:
        sys.stdout = saved

    stream.seek(0)
    lines = stream.readlines()
    col1, col2, col3 = lines[0].split(maxsplit=2)
    assert re.match('\s*PACKAGE\s*', col1)
    assert re.match('\s*AVAILABLE\s*', col2)
    assert re.match('\s*DESCRIPTION\s*', col3)
    previousPackage = None
    invalidCount = 0
    for line in lines[1:]:
        if not line or line == '\n':
            continue
        if line.startswith(' '):
            assert DEPENDENCIES[previousPackage].requires in line
            assert 'is required' in line
            invalidCount += 1
        else:
            package, available, description = line.split(maxsplit=2)
            assert package in DEPENDENCIES
            assert available == availability[package]
            assert description
            previousPackage = package
    if expectInvalid:
        assert invalidCount

def test_showAvailablePackages_current():
    availability = {}
    for pkg in DEPENDENCIES:
        try:
            importlib.import_module(pkg)
            availability[pkg] = "Yes"
        except ImportError:
            availability[pkg] = "No"
    if availability['keras'] == "No":
        availability['keras'] = availability['tensorflow']
    back_checkStatus(availability)

# just need key to be present, value is not used
mockInterfacesAvailable = {dep.name: None for dep in DEPENDENCIES.values()
                           if dep.section == 'interfaces'}

@patch(nimble._utility.DeferredModuleImport, 'nimbleAccessible', lambda self: True)
@patch(nimble.core.interfaces, 'available', mockInterfacesAvailable)
def test_showAvailablePackages_allYes():
    availability = {pkg: "Yes" for pkg in DEPENDENCIES}
    back_checkStatus(availability)

def raiseImportError(package):
    raise ImportError('no package')

def noInterfacesAvailable():
    nimble.core.interfaces.available = {}

@patch(nimble.core.configuration, 'checkVersion', raiseImportError)
@patch(nimble._utility.DeferredModuleImport, 'nimbleAccessible', lambda self: False)
@patch(nimble.core._learnHelpers, 'initAvailablePredefinedInterfaces',
       noInterfacesAvailable)
def test_showAvailablePackages_allNo():
    availability = {pkg: "No" for pkg in DEPENDENCIES}
    back_checkStatus(availability)

def raisePackageException(package):
    raise PackageException('version error')

@patch(nimble.core.configuration, 'checkVersion', raisePackageException)
@patch(nimble._utility.DeferredModuleImport, 'nimbleAccessible', raisePackageException)
@patch(nimble.core._learnHelpers, 'initAvailablePredefinedInterfaces',
       noInterfacesAvailable)
@patch(nimble.core.interfaces.universal_interface.PredefinedInterfaceMixin,
       '__init__', raisePackageException)
def test_showAvailablePackages_invalidIfInstalled():
    availability = {pkg: "No" for pkg in DEPENDENCIES}
    back_checkStatus(availability, expectInvalid=True)
