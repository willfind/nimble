"""
Tests to check the loading, writing, and usage of UML.settings, along
with the undlying structures being used.

"""

import tempfile
import copy
import os

from nose.tools import raises

import UML
from UML.exceptions import ArgumentException

def fileEqualObjOutput(fp, obj):
	resultFile = tempfile.TemporaryFile()
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


def testSCPCP_simple():
	""" Test that the ConfigParser subclass works with some simple data """
	fp = tempfile.TemporaryFile()
	template = makeDefaultTemplate()
	for line in template:
		fp.write(line)
	fp.seek(0)

	obj = UML.configuration.SortedCommentPreservingConfigParser()
	obj.readfp(fp)

	fileEqualObjOutput(fp, obj)

def testSCPCP_newOption():
	""" Test that comments are bound correctly after adding a new option """
	template = makeDefaultTemplate()

	fp = tempfile.TemporaryFile()
	for line in template:
		fp.write(line)
	fp.seek(0)

	obj = UML.configuration.SortedCommentPreservingConfigParser()
	obj.readfp(fp)

	obj.set("SectionName", "option2", '1')

	wanted = tempfile.TemporaryFile()
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

	fp = tempfile.TemporaryFile()
	for line in template:
		fp.write(line)
	fp.seek(0)

	obj = UML.configuration.SortedCommentPreservingConfigParser()
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

	fpWanted = tempfile.TemporaryFile()
	for line in templateWanted:
		fpWanted.write(line)
	fpWanted.seek(0)

	fpSpaced = tempfile.TemporaryFile()
	for line in templateSpaced:
		fpSpaced.write(line)
	fpSpaced.seek(0)

	obj = UML.configuration.SortedCommentPreservingConfigParser()
	obj.readfp(fpSpaced)
	fpSpaced.seek(0)

	# should be equal
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

	# confirm that changes is empty
	assert UML.settings.changes == origChangeSet



def test_settings_saving():
	""" Test UML.settings will save its in memory changes """
	# back up configuration.ini
	backup = backupConfigurationFile()

	try:
		# make some change via UML.settings. save it,
		UML.settings.set("bogusSectionName", "bogus.Option.Name", '1')
		UML.settings.saveChanges()

		# reload it with the starup function, make sure settings saved.
		UML.settings = UML.configuration.loadSettings()
		assert UML.settings.get("bogusSectionName", 'bogus.Option.Name') == '1'
	finally:
		# copy backup file over
		copyBackupOverConfigurationFile(backup)


def test_settings_syncing():
	""" Test UML.configuration.syncWithInteraces correctly modifies file """
	# back up configuration
	backup = backupConfigurationFile()
	backupAvailable = copy.copy(UML.interfaces.available)

	try:
		tempInterface = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
		UML.interfaces.available.append(tempInterface)

		# run sync
		UML.configuration.syncWithInteraces(UML.settings)

		# reload settings
		UML.settings = UML.configuration.loadSettings()

		# make sure new section and name was correctly added
		# '' is default value when adding options from interfaces
		assert UML.settings.get('Test', 'Temp0') == ''
		assert UML.settings.get('Test', 'Temp1') == ''
	finally:
		# restore available interfaces
		UML.interfaces.available = backupAvailable
		# retore config file
		copyBackupOverConfigurationFile(backup)
		# have to reload UML.settings too
		UML.settings = UML.configuration.loadSettings()


# test that when you sync an interface, it doesn't wipe out the values
# you already have in place
def test_settings_syncingSafety():
	""" Test that syncing preserves values already in the config file """
	# back up configuration
	backup = backupConfigurationFile()
	backupAvailable = copy.copy(UML.interfaces.available)

	try:
		tempInterface1 = OptionNamedLookalike("Test", ['Temp0', 'Temp1'])
		UML.interfaces.available.append(tempInterface1)

		# run sync, then reload
		UML.configuration.syncWithInteraces(UML.settings)
		UML.settings = UML.configuration.loadSettings()

		UML.settings.set('Test', 'Temp0', '0')
		UML.settings.set('Test', 'Temp1', '1')
		UML.settings.saveChanges()

		# now set up another trigger for syncing
		tempInterface2 = OptionNamedLookalike("TestOther", ['Temp0'])
		UML.interfaces.available.append(tempInterface2)

		# run sync, then reload
		UML.configuration.syncWithInteraces(UML.settings)
		UML.settings = UML.configuration.loadSettings()
		
		assert UML.settings.get("Test", 'Temp0') == '0'
		assert UML.settings.get("Test", 'Temp1') == '1'
	finally:
		UML.interfaces.available = backupAvailable
		copyBackupOverConfigurationFile(backup)
		UML.settings = UML.configuration.loadSettings()


@raises(ArgumentException)
def test_settings_allowedNames():
	""" Test that you can only set allowed names in interface sections """

	assert UML.settings.changes == {}
	UML.settings.set('Custom', 'Hello', "Goodbye")
	UML.settings.changes = {}



#test register and deregister custom learner?


###############
### Helpers ###
###############


class OptionNamedLookalike(object):
	def __init__(self, name, optNames):
		self.name = name
		self.optionNames = optNames
		
	def getCanonicalName(self):
		return self.name


def backupConfigurationFile():
	backup = tempfile.TemporaryFile()
	configurationFile = open(os.path.join(UML.UMLPath, 'configuration.ini'), 'r')
	backup.write(configurationFile.read())
	configurationFile.close()
	return backup

def copyBackupOverConfigurationFile(backup):
	backup.seek(0)
	configurationFile = open(os.path.join(UML.UMLPath, 'configuration.ini'), 'w')
	configurationFile.write(backup.read())
