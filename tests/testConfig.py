"""
Tests to check the loading, writing, and usage of UML.settings, along
with the undlying structures being used.
"""

from __future__ import absolute_import
import tempfile
import copy
import os

from nose.tools import raises
import six.moves.configparser

import UML
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue
from UML.exceptions import ImproperObjectAction
from UML.configuration import configSafetyWrapper


def fileEqualObjOutput(fp, obj):
    """
    fp must be readable
    """
    resultFile = tempfile.NamedTemporaryFile('w')
    obj.write(resultFile)

    fp.seek(0)
    resultFile.seek(0)

    origRet = fp.read()
    resultFile = open(resultFile.name, 'r')
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


def testSCPCP_simple():
    """ Test that the ConfigParser subclass works with some simple data """
    fp = tempfile.NamedTemporaryFile('w')
    template = makeDefaultTemplate()
    for line in template:
        fp.write(line)
    fp.seek(0)

    obj = UML.configuration.SortedCommentPreservingConfigParser()
    fp = open(fp.name, 'r')
    obj.readfp(fp)

    fileEqualObjOutput(fp, obj)


def testSCPCP_newOption():
    """ Test that comments are bound correctly after adding a new option """
    template = makeDefaultTemplate()

    fp = tempfile.NamedTemporaryFile('w')
    for line in template:
        fp.write(line)
    fp.seek(0)

    obj = UML.configuration.SortedCommentPreservingConfigParser()
    fp = open(fp.name, 'r')
    obj.readfp(fp)

    obj.set("SectionName", "option2", '1')

    wanted = tempfile.NamedTemporaryFile('w')
    template = makeDefaultTemplate()
    template.insert(8, "option2 = 1\n")
    for line in template:
        wanted.write(line)
    wanted.seek(0)

    wanted = open(wanted.name, 'r')
    fileEqualObjOutput(wanted, obj)


def testSCPCP_multilineComments():
    """ Test that multiline comments are preserved """
    template = makeDefaultTemplate()
    template.insert(5, "#SectionComment line 2\n")
    template.insert(6, "; Another comment, after an empty line\n")

    fp = tempfile.NamedTemporaryFile('w')
    for line in template:
        fp.write(line)
    fp.seek(0)

    obj = UML.configuration.SortedCommentPreservingConfigParser()
    fp = open(fp.name, 'r')
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

    obj = UML.configuration.SortedCommentPreservingConfigParser()
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


def test_settings_GetSet():
    """ Test UML.settings getters and setters """
    #orig changes
    origChangeSet = copy.deepcopy(UML.settings.changes)

    # for available interfaces
    for interface in UML.interfaces.available:
        name = interface.getCanonicalName()
        for option in interface.optionNames:
            # get values of options
            origValue = UML.settings.get(name, option)

            temp = "TEMPVALUE:" + name + option
            # change those values via UML.settings -
            UML.settings.set(name, option, temp)
            # check the change is reflected by all getters
            assert interface.getOption(option) == temp
            assert UML.settings.get(name, option) == temp

            # change it back
            interface.setOption(option, origValue)
            # check again
            assert UML.settings.get(name, option) == origValue

    # confirm that changes is the same
    assert UML.settings.changes == origChangeSet


@raises(InvalidArgumentType)
@configSafetyWrapper
def test_settings_HooksException_unCallable():
    """ Test SessionConfiguration.hook() throws exception on bad input """
    UML.settings.hook("TestS", "TestOp", 5)


@raises(ImproperObjectAction)
@configSafetyWrapper
def test_settings_HooksException_unHookable():
    """ Test SessionConfiguration.hook() throws exception for unhookable combo """
    UML.settings.hook("TestS", "TestOp", None)

    def nothing(value):
        pass

    UML.settings.hook("TestS", "TestOp", nothing)


@raises(InvalidArgumentValue)
@configSafetyWrapper
def test_settings_HooksException_wrongSig():
    """ Test SessionConfiguration.hook() throws exception on incorrect signature """
    def oneArg(value):
        pass

    UML.settings.hook("TestS", "TestOp", oneArg)

    def twoArg(value, value2):
        pass

    UML.settings.hook("TestS", "TestOp", twoArg)


@configSafetyWrapper
def test_settings_Hooks():
    """ Test the on-change hooks for a SessionConfiguration object """
    history = []

    def appendToHistory(newValue):
        history.append(newValue)

    UML.settings.hook("TestS", "TestOp", appendToHistory)

    UML.settings.set("TestS", "TestOp", 5)
    UML.settings.set("TestS", "TestOp", 4)
    UML.settings.set("TestS", "TestOp", 1)
    UML.settings.set("TestS", "TestOp", "Bang")

    assert history == [5, 4, 1, "Bang"]


@configSafetyWrapper
def test_settings_GetSectionOnly():
    """ Test UML.settings.get when only specifying a section """
    UML.settings.set("TestSec1", "op1", '1')
    UML.settings.set("TestSec1", "op2", '2')

    allSec1 = UML.settings.get("TestSec1", None)
    assert allSec1["op1"] == '1'
    assert allSec1['op2'] == '2'


#@configSafetyWrapper
#def test_settings_getFormatting():
#	""" Test the format flags  """
#	UML.settings.set("FormatTest", "numOp", 1)
#	asInt = UML.settings.get("FormatTest", "numOp", asFormat='int')
#	asFloat = UML.settings.get("FormatTest", "numOp", asFormat='float')

#	assert asInt == 1
#	assert asFloat == 1.0


@configSafetyWrapper
def test_settings_saving():
    """ Test UML.settings will save its in memory changes """
    # make some change via UML.settings. save it,
    UML.settings.set("newSectionName", "new.Option.Name", '1')
    UML.settings.saveChanges()

    # reload it with the starup function, make sure settings saved.
    UML.settings = UML.configuration.loadSettings()
    assert UML.settings.get("newSectionName", 'new.Option.Name') == '1'


@configSafetyWrapper
def test_settings_savingSection():
    """ Test UML.settings.saveChanges when specifying a section """
    UML.settings.set("TestSec1", "op1", '1')
    UML.settings.set("TestSec1", "op2", '2')
    UML.settings.set("TestSec2", "op1", '1')
    UML.settings.saveChanges("TestSec1")

    # assert that other changes are still in effect
    assert len(UML.settings.changes) == 1
    assert UML.settings.get("TestSec2", "op1") == '1'

    # reload it with the starup function, make sure settings saved.
    temp = UML.configuration.loadSettings()
    assert temp.get('TestSec1', "op1") == '1'
    assert temp.get('TestSec1', "op2") == '2'
    # confirm that the change outside the section was not saved
    try:
        val = temp.get('TestSec2', "op1")
        assert False
    except six.moves.configparser.NoSectionError:
        pass


@configSafetyWrapper
def test_settings_savingOption():
    """ Test UML.settings.saveChanges when specifying a section and option """
    UML.settings.set("TestSec1", "op1", '1')
    UML.settings.set("TestSec1", "op2", '2')
    UML.settings.set("TestSec2", "op1", '1')
    UML.settings.saveChanges("TestSec1", "op2")

    # assert that other changes are still in effect
    assert len(UML.settings.changes["TestSec1"]) == 1
    assert len(UML.settings.changes["TestSec2"]) == 1
    assert len(UML.settings.changes) == 2
    assert UML.settings.get("TestSec2", "op1") == '1'
    assert UML.settings.get("TestSec1", "op1") == '1'

    # reload it with the starup function, make that option was saved.
    temp = UML.configuration.loadSettings()
    assert temp.get('TestSec1', "op2") == '2'
    # confirm that the other changes were not saved
    try:
        val = temp.get('TestSec2', "op1")
        assert False
    except six.moves.configparser.NoSectionError:
        pass
    try:
        val = temp.get('TestSec1', "op1") == '1'
        assert False
    except six.moves.configparser.NoOptionError:
        pass


@configSafetyWrapper
def test_settings_syncingNewInterface():
    """ Test UML.configuration.syncWithInterfaces correctly modifies file """
    tempInterface = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
    UML.interfaces.available.append(tempInterface)
    ignoreInterface = OptionNamedLookalike("ig", [])
    UML.interfaces.available.append(ignoreInterface)

    # run sync
    UML.configuration.syncWithInterfaces(UML.settings)

    # reload settings - to make sure the syncing was recorded
    UML.settings = UML.configuration.loadSettings()

    # make sure there is no section associated with the optionless
    # interface
    assert not UML.settings.cp.has_section('ig')

    # make sure new section and name was correctly added
    # '' is default value when adding options from interfaces
    assert UML.settings.get('Test', 'Temp0') == ''
    assert UML.settings.get('Test', 'Temp1') == ''


@configSafetyWrapper
def test_settings_syncingSafety():
    """ Test that syncing preserves values already in the config file """
    tempInterface1 = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
    UML.interfaces.available.append(tempInterface1)

    # run sync, then reload
    UML.configuration.syncWithInterfaces(UML.settings)
    UML.settings = UML.configuration.loadSettings()

    UML.settings.set('Test', 'Temp0', '0')
    UML.settings.set('Test', 'Temp1', '1')
    UML.settings.saveChanges()

    # now set up another trigger for syncing
    tempInterface2 = OptionNamedLookalike("TestOther", ['Temp0'])
    UML.interfaces.available.append(tempInterface2)

    # run sync, then reload
    UML.configuration.syncWithInterfaces(UML.settings)
    UML.settings = UML.configuration.loadSettings()

    assert UML.settings.get("Test", 'Temp0') == '0'
    assert UML.settings.get("Test", 'Temp1') == '1'


@configSafetyWrapper
def test_settings_syncingChanges():
    """ Test that syncing interfaces properly saves current changes """
    tempInterface1 = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
    tempInterface2 = OptionNamedLookalike("TestOther", ['Temp0'])
    UML.interfaces.available.append(tempInterface1)
    UML.interfaces.available.append(tempInterface2)

    # run sync, then reload
    UML.configuration.syncWithInterfaces(UML.settings)
    UML.settings = UML.configuration.loadSettings()

    UML.settings.set('Test', 'Temp0', '0')
    UML.settings.set('Test', 'Temp1', '1')
    UML.settings.set('TestOther', 'Temp0', 'unchanged')

    assert UML.settings.get('Test', 'Temp0') == '0'

    # change Test option names and resync
    tempInterface1.optionNames[1] = 'NotTemp1'
    UML.configuration.syncWithInterfaces(UML.settings)

    # check values of both changed and unchanged names
    assert UML.settings.get('Test', 'Temp0') == '0'
    try:
        UML.settings.get('Test', 'Temp1')
    except six.moves.configparser.NoOptionError:
        pass
    assert UML.settings.get('Test', 'NotTemp1') == ''

    # check that the temp value for testOther is unaffeected
    assert UML.settings.get('TestOther', 'Temp0') == 'unchanged'


@raises(InvalidArgumentValue)
def test_settings_allowedNames():
    """ Test that you can only set allowed names in interface sections """

    assert UML.settings.changes == {}
    UML.settings.set('Custom', 'Hello', "Goodbye")
    UML.settings.changes = {}


@configSafetyWrapper
@raises(six.moves.configparser.NoSectionError)
# test that set witout save is temporary
def test_settings_set_without_save1():
    # make some change via UML.settings.
    UML.settings.set("tempSectionName", "temp.Option.Name", '1')

    UML.settings.get("tempSectionName", 'temp.Option.Name') == '1'

    # reload it with the starup function, try to load something which
    # shouldn't be there
    UML.settings = UML.configuration.loadSettings()
    UML.settings.get("tempSectionName", 'temp.Option.Name')


@configSafetyWrapper
@raises(six.moves.configparser.NoSectionError)
# test that set witout save is temporary
def test_settings_set_without_save2():
    # make some change via UML.settings.
    UML.settings.set("tempSectionName", "temp.Option.Name", '1')
    assert UML.settings.get("tempSectionName", 'temp.Option.Name') == '1'

    # reload it with the starup function, try to load something which
    # shouldn't be there
    UML.settings = UML.configuration.loadSettings()
    UML.settings.get("tempSectionName", 'temp.Option.Name')


@configSafetyWrapper
# test that delete then save will change file - value
def test_settings_deleteThenSaveAValue():
    UML.settings.set("tempSectionName", "temp.Option.Name1", '1')
    UML.settings.set("tempSectionName", "temp.Option.Name2", '2')
    assert UML.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    UML.settings.saveChanges()

    # change reflected in memory
    UML.settings.delete("tempSectionName", "temp.Option.Name1")
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except six.moves.configparser.NoOptionError:
        pass
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    # change isn't reflected in file
    UML.settings = UML.configuration.loadSettings()
    # previous delete wasn't saved, so this should still work
    assert UML.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'
    UML.settings.delete("tempSectionName", "temp.Option.Name1")
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except six.moves.configparser.NoOptionError:
        pass

    UML.settings.saveChanges()

    # change should now be reflected in file
    UML.settings = UML.configuration.loadSettings()
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except six.moves.configparser.NoOptionError:
        pass
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'


@configSafetyWrapper
# test that delete then save will change file - section
def test_settings_deleteThenSaveASection():
    UML.settings.set("tempSectionName", "temp.Option.Name1", '1')
    UML.settings.set("tempSectionName", "temp.Option.Name2", '2')
    assert UML.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'
    UML.settings.saveChanges()

    # change reflected in memory
    UML.settings.delete("tempSectionName", None)
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name2')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass

    # change isn't reflected in file
    UML.settings = UML.configuration.loadSettings()
    # previous delete wasn't saved, so this should still work
    assert UML.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'
    UML.settings.delete("tempSectionName", None)
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name2')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass

    UML.settings.saveChanges()

    # change should now be reflected in file
    UML.settings = UML.configuration.loadSettings()
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name2')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass


@configSafetyWrapper
# test that deleteing an unsaved set is a cycle - value
def test_settings_setThenDeleteCycle_value():
    UML.settings.set("tempSectionName", "temp.Option.Name1", '1')
    UML.settings.set("tempSectionName", "temp.Option.Name2", '2')
    assert UML.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    # change reflected in memory
    UML.settings.delete("tempSectionName", "temp.Option.Name1")
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except six.moves.configparser.NoOptionError:
        pass

    # change should now be reflected in file
    UML.settings = UML.configuration.loadSettings()
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass


@configSafetyWrapper
# test that deleteing an unsaved set is a cycle - section
def test_settings_setThenDeleteCycle_section():
    UML.settings.set("tempSectionName", "temp.Option.Name1", '1')
    UML.settings.set("tempSectionName", "temp.Option.Name2", '2')
    assert UML.settings.get("tempSectionName", 'temp.Option.Name1') == '1'

    # change reflected in memory
    UML.settings.delete("tempSectionName", None)
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass

    # change never saved, shouldn't be in file
    UML.settings = UML.configuration.loadSettings()
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass


@configSafetyWrapper
def test_settings_setDefault():
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name2')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass

    UML.settings.set("tempSectionName", "temp.Option.Name1", '1')
    UML.settings.setDefault("tempSectionName", "temp.Option.Name2", '2')
    assert UML.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    # Name2 should be reflected in file, but not Name1
    UML.settings = UML.configuration.loadSettings()
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except six.moves.configparser.NoOptionError:
        pass

    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'


@configSafetyWrapper
def test_settings_deleteDefault():
    UML.settings.setDefault("tempSectionName", "temp.Option.Name1", '1')
    UML.settings.setDefault("tempSectionName", "temp.Option.Name2", '2')
    assert UML.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    # Establish a baseline
    UML.settings = UML.configuration.loadSettings()
    assert UML.settings.get("tempSectionName", 'temp.Option.Name1') == '1'
    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    UML.settings.deleteDefault("tempSectionName", 'temp.Option.Name1')

    UML.settings = UML.configuration.loadSettings()
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoOptionError
    except six.moves.configparser.NoOptionError:
        pass

    assert UML.settings.get("tempSectionName", 'temp.Option.Name2') == '2'

    UML.settings.deleteDefault("tempSectionName", None)
    UML.settings = UML.configuration.loadSettings()
    try:
        UML.settings.get("tempSectionName", 'temp.Option.Name1')
        assert False  # expected ConfigParser.NoSectionError
    except six.moves.configparser.NoSectionError:
        pass


def testToDeleteSentinalObject():
    val = UML.configuration.ToDelete()

    assert isinstance(val, UML.configuration.ToDelete)


###############
### Helpers ###
###############


class OptionNamedLookalike(object):
    def __init__(self, name, optNames):
        self.name = name
        self.optionNames = optNames

    def getCanonicalName(self):
        return self.name
