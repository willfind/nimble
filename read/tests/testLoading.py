"""
Unit tests for functions that get lists of files/directories from a provided directory,
found in read.data_loading module
"""

from UML.read.data_loading import *
import UML
import os

testDirectory = os.path.join(UML.UMLPath, 'read', 'tests', 'testDirectory')

def test_listFiles():
	"""
	Unit test for listFiles function in read.data_loading module
	"""
	print testDirectory

	fileList = [fileName.lower() for fileName in listFiles(testDirectory)]
	print fileList

	assert (os.path.join(testDirectory, '001.txt')).lower() in fileList
	assert (os.path.join(testDirectory, '002.txt')).lower() in fileList
	assert (os.path.join(testDirectory, '001.html')).lower() in fileList
	assert (os.path.join(testDirectory, '050.html')).lower() in fileList
	assert (os.path.join(testDirectory, 'body')).lower() not in fileList
	assert (os.path.join(testDirectory, 'body/')).lower() not in fileList

def test_listDirs():
	"""
	Unit test for listDirs function in read.data_loading module
	"""
	dirList = [dirName.lower() for dirName in listDirs(testDirectory)]
	assert (os.path.join(testDirectory, '001.txt')).lower() not in dirList
	assert (os.path.join(testDirectory, '002.txt')).lower() not in dirList
	assert (os.path.join(testDirectory, '001.html')).lower() not in dirList
	assert (os.path.join(testDirectory, '050.html')).lower() not in dirList
	assert ((os.path.join(testDirectory, 'body')).lower() in dirList) or ((os.path.join(testDirectory, 'body/')).lower() in dirList)

def test_dirMapper():
	"""
	Unit test for dirMapper function in read.data_loading module
	"""
	fileList = dirMapper(testDirectory, ['.txt'], 'multiTyped')

	assert '001' in fileList
	assert '002' in fileList
	assert '003' in fileList
	assert '004' in fileList
	assert '005' in fileList
	assert '006' not in fileList
	assert '007' in fileList
	assert '008' in fileList
	assert '009' in fileList

	innerDictOne = fileList['001']
	innerDictTwo = fileList['002']
	innerDictFive = fileList['005']


	assert 'head' not in innerDictOne
	assert 'body' in innerDictOne
	assert 'head' in innerDictTwo
	assert 'body' in innerDictTwo
	assert 'head' in innerDictFive
	assert 'body' not in innerDictFive

	oneFileListBody = [fileName.lower() for fileName in innerDictOne['body']]
	twoFileListHead = [fileName.lower() for fileName in innerDictTwo['head']]
	twoFileListBody = [fileName.lower() for fileName in innerDictTwo['body']]

	assert (os.path.join(testDirectory, 'body/001.txt')).lower() in oneFileListBody
	assert (os.path.join(testDirectory, 'head/002.txt')).lower() in twoFileListHead
	assert (os.path.join(testDirectory, 'body/002.txt')).lower() in twoFileListBody
	assert (os.path.join(testDirectory, 'body/a/002.txt')).lower() in twoFileListBody
#	assert (os.path.join(testDirectory, 'body/b/008.txt')).lower() in eightFileListBody

def test_recursiveFileLister():
	"""
	Unit test of the recursiveFileLister function in read.data_loading module
	"""
	textFileList = [fileName.lower() for fileName in recursiveFileLister(testDirectory, ['.txt'])]

	assert (os.path.join(testDirectory, 'body/001.txt')).lower() in textFileList
	assert (os.path.join(testDirectory, 'body/002.txt')).lower() in textFileList
	assert (os.path.join(testDirectory, 'body/003.txt')).lower() in textFileList
	assert (os.path.join(testDirectory, 'body/004.txt')).lower() in textFileList
	assert (os.path.join(testDirectory, '001.txt')).lower() in textFileList
	assert (os.path.join(testDirectory, '002.txt')).lower() in textFileList
	assert (os.path.join(testDirectory, 'body/a/001.txt')).lower() in textFileList
	assert (os.path.join(testDirectory, 'body/b/007.txt')).lower() in textFileList

	allFileList = [fileName.lower() for fileName in recursiveFileLister(testDirectory)]

	assert (os.path.join(testDirectory, 'body/001.txt')).lower() in allFileList
	assert (os.path.join(testDirectory, 'body/002.txt')).lower() in allFileList
	assert (os.path.join(testDirectory, 'body/003.txt')).lower() in allFileList
	assert (os.path.join(testDirectory, 'body/004.txt')).lower() in allFileList
	assert (os.path.join(testDirectory, '001.txt')).lower() in allFileList
	assert (os.path.join(testDirectory, '002.txt')).lower() in allFileList
	assert (os.path.join(testDirectory, 'body/a/001.txt')).lower() in allFileList
	assert (os.path.join(testDirectory, 'body/b/007.txt')).lower() in allFileList
	assert (os.path.join(testDirectory, '001.html')).lower() in allFileList
	assert (os.path.join(testDirectory, '050.html')).lower() in allFileList
	assert (os.path.join(testDirectory, '001.savv')).lower() in allFileList

def test_extractFilename():
	"""
	Unit test of the extractFilename function in read.data_loading module
	"""
	testFilePath1 = '/blard/flard/nard/dog.txt'
	testFilename1 = extractFilename(testFilePath1)
	assert testFilename1 == 'dog'

	testFilePath2 = 'dog.txt.html'
	testFilename2 = extractFilename(testFilePath2)
	assert testFilename2 == 'dog.txt'
	
