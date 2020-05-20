"""
Tests to check the loading, writing, and usage of nimble.settings, along
with the undlying structures being used.
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
from nimble.interfaces.universal_interface import UniversalInterface
from nimble.interfaces.universal_interface import PredefinedInterface
from .assertionHelpers import configSafetyWrapper


###############
### Helpers ###
###############

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
        return name.lower() == cls.getCanonicalName()

    @classmethod
    def provideInitExceptionInfo(cls):
        raise PackageException("failed to load interface")


def fileEqualObjOutput(fp, obj):
    """
    fp must be readable
    """
    with tempfile.NamedTemporaryFile('w+') as resultFile:
        obj.write(resultFile)

        fp.seek(0)
        resultFile.seek(0)

        origRet = fp.read()
        objRet = resultFile.read()

        assert origRet == objRet


def makeDefaultTemplate():
    lines = [None] * 11
    # Note: the formatting and ordering must be the same as how the
    # ConfigParser outputs them normally
    lines[0] = "#Now, defaults:\n"
    lines[1] = "[DEFAULT]\n"
    lines[2] = "superkey = 44\n"
    lines[3] = "\n"
    lines[4] = "#section comment\n"
    lines[5] = "[SectionName]\n"
    lines[6] = "#option comment\n"
    lines[7] = "option1 = 5\n"
    lines[8] = "#between 1 and 3 originally\n"
    lines[9] = "option3 = 3\n"
    lines[10] = "\n"  # ConfigParser always writes two newlines at the end

    return lines


#############
### Tests ###
#############


def testSCPCP_simple():
    """ Test that the ConfigParser subclass works with some simple data """
    with tempfile.NamedTemporaryFile('w') as fp:
        template = makeDefaultTemplate()
        for line in template:
            fp.write(line)
        fp.seek(0)

        obj = nimble.configuration.SortedCommentPreservingConfigParser()
        with open(fp.name, 'r') as fp:
            obj.readfp(fp)

            fileEqualObjOutput(fp, obj)



def testSCPCP_newOption():
    """ Test that comments are bound correctly after adding a new option """
    template = makeDefaultTemplate()

    with tempfile.NamedTemporaryFile('w+') as fp:
        for line in template:
            fp.write(line)
        fp.seek(0)

        obj = nimble.configuration.SortedCommentPreservingConfigParser()
        with open(fp.name, 'r') as fp:
            obj.readfp(fp)

            obj.set("SectionName", "option2", '1')

            with tempfile.NamedTemporaryFile('w+') as wanted:
                template = makeDefaultTemplate()
                template.insert(8, "option2 = 1\n")
                for line in template:
                    wanted.write(line)
                wanted.seek(0)

                fileEqualObjOutput(wanted, obj)


def testSCPCP_multilineComments():
    """ Test that multiline comments are preserved """
    template = makeDefaultTemplate()
    template.insert(5, "#SectionComment line 2\n")
    template.insert(6, "; Another comment, after an empty line\n")

    with tempfile.NamedTemporaryFile('w') as fp:
        for line in template:
            fp.write(line)
        fp.seek(0)

        obj = nimble.configuration.SortedCommentPreservingConfigParser()
        with open(fp.name, 'r') as fp:
            obj.readfp(fp)

            fp.seek(0)
            fileEqualObjOutput(fp, obj)


def testSCPCP_whitespaceIgnored():
    """ Test that white space between comment lines is ignored """
    templateWanted = makeDefaultTemplate()
    templateSpaced = makeDefaultTemplate()

    templateWanted.insert(5, "#SectionComment line 2\n")
    templateWanted.insert(6, "; Another comment, after an empty line\n")

    templateSpaced.insert(5, "#SectionComment line 2\n")
    templateSpaced.insert(6, "\n")
    templateSpaced.insert(7, "; Another comment, after an empty line\n")

    fpWanted = tempfile.NamedTemporaryFile('w')
    for line in templateWanted:
        fpWanted.write(line)
    fpWanted.seek(0)

    fpSpaced = tempfile.NamedTemporaryFile('w')
    for line in templateSpaced:
        fpSpaced.write(line)
    fpSpaced.seek(0)

    obj = nimble.configuration.SortedCommentPreservingConfigParser()
    fpSpaced = open(fpSpaced.name, 'r')
    obj.readfp(fpSpaced)
    fpSpaced.seek(0)

    # should be equal
    fpWanted = open(fpWanted.name, 'r')
    fileEqualObjOutput(fpWanted, obj)

    # should raise Assertion error
    try:
        fileEqualObjOutput(fpSpaced, obj)
    except AssertionError:
        pass

    fpWanted.close()
    fpSpaced.close()

def test_settings_GetSet():
    """ Test nimble.settings getters and setters """
    #orig changes
    origChangeSet = copy.deepcopy(nimble.settings.changes)

    # for available interfaces
    for interface in nimble.interfaces.available.values():
        name = interface.getCanonicalName()
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
            # check again
            assert nimble.settings.get(name, option) == origValue

    # confirm that changes is the same
    assert nimble.settings.changes == origChangeSet


@raises(InvalidArgumentType)
@configSafetyWrapper
def test_settings_HooksException_unCallable():
    """ Test SessionConfiguration.hook() throws exception on bad input """
    nimble.settings.hook("TestS", "TestOp", 5)


@raises(ImproperObjectAction)
@configSafetyWrapper
def test_settings_HooksException_unHookable():
    """ Test SessionConfiguration.hook() throws exception for unhookable combo """
    nimble.settings.hook("TestS", "TestOp", None)

    def nothing(value):
        pass

    nimble.settings.hook("TestS", "TestOp", nothing)


@raises(InvalidArgumentValue)
@configSafetyWrapper
def test_settings_HooksException_wrongSig():
    """ Test SessionConfiguration.hook() throws exception on incorrect signature """
    def twoArg(value, value2):
        pass

    nimble.settings.hook("TestS", "TestOp", twoArg)


@configSafetyWrapper
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


@configSafetyWrapper
def test_settings_GetSectionOnly():
    """ Test nimble.settings.get when only specifying a section """
    nimble.settings.set("TestSec1", "op1", '1')
    nimble.settings.set("TestSec1", "op2", '2')

    allSec1 = nimble.settings.get("TestSec1", None)
    assert allSec1["op1"] == '1'
    assert allSec1['op2'] == '2'


#@configSafetyWrapper
#def test_settings_getFormatting():
#	""" Test the format flags  """
#	nimble.settings.set("FormatTest", "numOp", 1)
#	asInt = nimble.settings.get("FormatTest", "numOp", asFormat='int')
#	asFloat = nimble.settings.get("FormatTest", "numOp", asFormat='float')

#	assert asInt == 1
#	assert asFloat == 1.0


@configSafetyWrapper
def test_settings_saving():
    """ Test nimble.settings will save its in memory changes """
    # make some change via nimble.settings. save it,
    nimble.settings.set("newSectionName", "new.Option.Name", '1')
    nimble.settings.saveChanges()

    # reload it with the starup function, make sure settings saved.
    nimble.settings = nimble.configuration.loadSettings()
    assert nimble.settings.get("newSectionName", 'new.Option.Name') == '1'


@configSafetyWrapper
def test_settings_savingSection():
    """ Test nimble.settings.saveChanges when specifying a section """
    nimble.settings.set("TestSec1", "op1", '1')
    nimble.settings.set("TestSec1", "op2", '2')
    nimble.settings.set("TestSec2", "op1", '1')
    nimble.settings.saveChanges("TestSec1")

    # assert that other changes are still in effect
    assert len(nimble.settings.changes) == 1
    assert nimble.settings.get("TestSec2", "op1") == '1'

    # reload it with the starup function, make sure settings saved.
    temp = nimble.configuration.loadSettings()
    assert temp.get('TestSec1', "op1") == '1'
    assert temp.get('TestSec1', "op2") == '2'
    # confirm that the change outside the section was not saved
    try:
        val = temp.get('TestSec2', "op1")
        assert False
    except configparser.NoSectionError:
        pass


@configSafetyWrapper
def test_settings_savingOption():
    """ Test nimble.settings.saveChanges when specifying a section and option """
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
    temp = nimble.configuration.loadSettings()
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


@configSafetyWrapper
def test_settings_addingNewInterface():
    """ Test nimble.configuration.setInterfaceOptions correctly modifies file """
    tempInterface = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
    nimble.interfaces.available[tempInterface.name] = tempInterface
    ignoreInterface = OptionNamedLookalike("ig", [])
    nimble.interfaces.available[ignoreInterface.name] = ignoreInterface

    # set options for all interfaces
    for interface in nimble.interfaces.available.values():
        nimble.configuration.setInterfaceOptions(nimble.settings, interface, True)

    # reload settings - to make sure the options setting was recorded
    nimble.settings = nimble.configuration.loadSettings()

    # make sure there is no section associated with the optionless
    # interface
    assert not nimble.settings.cp.has_section('ig')

    # make sure new section and name was correctly added
    # '' is default value when adding options from interfaces
    assert nimble.settings.get('Test', 'Temp0') == ''
    assert nimble.settings.get('Test', 'Temp1') == ''


@configSafetyWrapper
def test_settings_setInterfaceOptionsSafety():
    """ Test that setting options preserves values already in the config file """
    tempInterface1 = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
    nimble.interfaces.available[tempInterface1.name] = tempInterface1

    # set options for all interfaces, then reload
    for interface in nimble.interfaces.available.values():
        nimble.configuration.setInterfaceOptions(nimble.settings, interface, True)
    nimble.settings = nimble.configuration.loadSettings()

    nimble.settings.set('Test', 'Temp0', '0')
    nimble.settings.set('Test', 'Temp1', '1')
    nimble.settings.saveChanges()

    # now set up another trigger to set options for
    tempInterface2 = OptionNamedLookalike("TestOther", ['Temp0'])
    nimble.interfaces.available[tempInterface2.name] = tempInterface2

    # set options for all interfaces, then reload
    for interface in nimble.interfaces.available.values():
        nimble.configuration.setInterfaceOptions(nimble.settings, interface, True)
    nimble.settings = nimble.configuration.loadSettings()

    assert nimble.settings.get("Test", 'Temp0') == '0'
    assert nimble.settings.get("Test", 'Temp1') == '1'


@configSafetyWrapper
def test_settings_setInterfaceOptionsChanges():
    """ Test that setting interface options properly saves current changes """
    tempInterface1 = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
    tempInterface2 = OptionNamedLookalike("TestOther", ['Temp0'])
    nimble.interfaces.available[tempInterface1.name] = tempInterface1
    nimble.interfaces.available[tempInterface2.name] = tempInterface2

    # set options for all interfaces, then reload
    nimble.configuration.setAndSaveAvailableInterfaceOptions()
    nimble.settings = nimble.configuration.loadSettings()

    nimble.settings.set('Test', 'Temp0', '0')
    nimble.settings.set('Test', 'Temp1', '1')
    nimble.settings.set('TestOther', 'Temp0', 'unchanged')

    assert nimble.settings.get('Test', 'Temp0') == '0'

    # change Test option names and reset options for all interfaces
    tempInterface1.optionNames[1] = 'NotTemp1'
    nimble.configuration.setAndSaveAvailableInterfaceOptions()

    # check values of both changed and unchanged names
    assert nimble.settings.get('Test', 'Temp0') == '0'
    try:
        nimble.settings.get('Test', 'Temp1')
        assert False
    except configparser.NoOptionError:
        pass
    assert nimble.settings.get('Test', 'NotTemp1') == ''

    # check that the temp value for testOther is unaffected
    assert nimble.settings.get('TestOther', 'Temp0') == 'unchanged'

@configSafetyWrapper
@raises(InvalidArgumentValue)
def test_settings_allowedNames():
    """ Test that you can only set allowed names in interface sections """

    assert nimble.settings.changes == {}
    nimble.settings.set('nimble', 'Hello', "Goodbye")
    nimble.settings.changes = {}


@configSafetyWrapper
@raises(configparser.NoSectionError)
# test that set without save is temporary
def test_settings_set_without_save():
    # make some change via nimble.settings.
    nimble.settings.set("tempSectionName", "temp.Option.Name", '1')
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name') == '1'

    # reload it with the startup function, try to load something which
    # shouldn't be there
    nimble.settings = nimble.configuration.loadSettings()
    nimble.settings.get("tempSectionName", 'temp.Option.Name')


@configSafetyWrapper
# test that delete then save will change file - value
def test_settings_deleteThenSaveAValue():
    nimble.settings.set("tempSectionName", "temp.Option.Name1", '1')
    nimble.settings.set("tempSectionName", "temp.Option.Name2", '2')
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    nimble.settings.saveChanges()

    # change reflected in memory
    nimble.settings.delete("tempSectionName", "temp.Option.Name1")
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except configparser.NoOptionError:
        pass
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    # change isn't reflected in file
    nimble.settings = nimble.configuration.loadSettings()
    # previous delete wasn't saved, so this should still work
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'
    nimble.settings.delete("tempSectionName", "temp.Option.Name1")
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except configparser.NoOptionError:
        pass

    nimble.settings.saveChanges()

    # change should now be reflected in file
    nimble.settings = nimble.configuration.loadSettings()
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except configparser.NoOptionError:
        pass
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'


@configSafetyWrapper
# test that delete then save will change file - section
def test_settings_deleteThenSaveASection():
    nimble.settings.set("tempSectionName", "temp.Option.Name1", '1')
    nimble.settings.set("tempSectionName", "temp.Option.Name2", '2')
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'
    nimble.settings.saveChanges()

    # change reflected in memory
    nimble.settings.delete("tempSectionName", None)
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name2')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass

    # change isn't reflected in file
    nimble.settings = nimble.configuration.loadSettings()
    # previous delete wasn't saved, so this should still work
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'
    nimble.settings.delete("tempSectionName", None)
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name2')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass

    nimble.settings.saveChanges()

    # change should now be reflected in file
    nimble.settings = nimble.configuration.loadSettings()
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name2')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass


@configSafetyWrapper
# test that deleteing an unsaved set is a cycle - value
def test_settings_setThenDeleteCycle_value():
    nimble.settings.set("tempSectionName", "temp.Option.Name1", '1')
    nimble.settings.set("tempSectionName", "temp.Option.Name2", '2')
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    # change reflected in memory
    nimble.settings.delete("tempSectionName", "temp.Option.Name1")
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except configparser.NoOptionError:
        pass

    # change should now be reflected in file
    nimble.settings = nimble.configuration.loadSettings()
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass


@configSafetyWrapper
# test that deleteing an unsaved set is a cycle - section
def test_settings_setThenDeleteCycle_section():
    nimble.settings.set("tempSectionName", "temp.Option.Name1", '1')
    nimble.settings.set("tempSectionName", "temp.Option.Name2", '2')
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name1') == '1'

    # change reflected in memory
    nimble.settings.delete("tempSectionName", None)
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass

    # change never saved, shouldn't be in file
    nimble.settings = nimble.configuration.loadSettings()
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass


@configSafetyWrapper
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
    nimble.settings = nimble.configuration.loadSettings()
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except configparser.NoOptionError:
        pass

    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'


@configSafetyWrapper
def test_settings_deleteDefault():
    nimble.settings.setDefault("tempSectionName", "temp.Option.Name1", '1')
    nimble.settings.setDefault("tempSectionName", "temp.Option.Name2", '2')
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    # Establish a baseline
    nimble.settings = nimble.configuration.loadSettings()
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    nimble.settings.deleteDefault("tempSectionName", 'temp.Option.Name1')

    nimble.settings = nimble.configuration.loadSettings()
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except configparser.NoOptionError:
        pass

    assert nimble.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    nimble.settings.deleteDefault("tempSectionName", None)
    nimble.settings = nimble.configuration.loadSettings()
    try:
        nimble.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except configparser.NoSectionError:
        pass


def testToDeleteSentinalObject():
    val = nimble.configuration.ToDelete()

    assert isinstance(val, nimble.configuration.ToDelete)


@configSafetyWrapper
@mock.patch('nimble.interfaces.predefined', [FailedPredefined])
def testSetLocationForFailedPredefinedInterface():
    nimble.settings.set('FailedPredefined', 'location', 'path/to/mock')


@configSafetyWrapper
@raises(InvalidArgumentValue)
@mock.patch('nimble.interfaces.predefined', [FailedPredefined])
def testExceptionSetOptionForFailedPredefinedInterface():
    nimble.settings.set('FailedPredefined', 'foo', 'path/to/mock')
