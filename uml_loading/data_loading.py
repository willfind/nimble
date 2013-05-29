import os

from UML.utility import ArgumentException

def dirMapper(dirPath, extensions=[], mode="multiTyped"):
	"""
	Return a map of {id:{type:[list of file paths]}}, where ids uniquely
	identify documents (not necessarily files). Id's are taken from fileNames; 
	each document should be uniquely identified their filename.  This function
	has two modes: 'oneType' and 'multiTyped'.  In oneType mode, all files
	in dirPath are treated as being of the same type - 

	Parameters:
		dirPath: path to the directory containing files to be processed

		extensions: list of file extensions (of format '.xxx') denoting
		which files to include in the returned list.  Any file whose extension
		is not contained within extensions will not be included in the results.
		Defaults all extensions, in which case all files will be included (but
		not directories).

		mode:  Either 'oneType' or 'multiTyped'.  In oneType mode, all files within
		dirPath or any subdirectory of dirPath are gathered, and all files with the
		same fileName (and any extension within the extensions list) will be appended
		to a list keyed by 'all' in the inner dictionary.  In multiTyped mode, a file's
		'type' is determined by which of dirPath's immediate subdirectories contains it.
		E.g.: if dirPath contains two subfolders, 'heading' and 'body', then all files
		within 'heading' or any subfolder would be grouped together, and all files within
		'body' or any subfolder would be grouped together.  multiTyped mode assumes that
		files are organized into subfolders of dirPath, so won't include files whose
		parent folder is dirPath (i.e. dirPath/001.txt would not be include in the 
		returned map of files, but dirPath/heading/001.txt and dirPath/body/001.txt would).
	"""
	results = {}
	#os.walk returns tuples with lists of all subdirectories and files contained
	#in the dirPath directory
	root = dirPath
	topDirs = listDirs(dirPath)

	if mode == "oneType":
		allFiles = recursiveFileLister(root, extensions)
		for filePath in allFiles:
			docId = extractFilename(filePath)
			if docId in results:
				fileList = results[docId]['all']
				fileList.append(filePath)
			else:
				fileList = [filePath]
				typeFileListMap = {'all':fileList}
				results[docId] = typeFileListMap
	elif mode == "multiTyped":
		for topDir in topDirs:
			typeName = os.path.basename(topDir)
			topDirPath = os.path.join(root, topDir)
			allFiles = recursiveFileLister(topDirPath, extensions)
			for filePath in allFiles:
				docId = extractFilename(filePath)
				if docId in results:
					typeFileListMap = results[docId]
					if typeName in typeFileListMap:
						fileList = typeFileListMap[typeName]
						fileList.append(filePath)
					else:
						fileList = [filePath]
						typeFileListMap[typeName] = fileList
				else:
					fileList = [filePath]
					typeFileListMap = {typeName:fileList}
					results[docId] = typeFileListMap

	return results


def recursiveFileLister(dirPath, extensions=[]):
	"""
	Get all files of a given extension contained within the provided directory and all
	of its subdirectories.  If no extension is provided, returns a list of all 
	(non-directory) files regardless of extension.  Full path of each file is provided.
	"""
	fileList = []
	walkList = os.walk(dirPath)
	for root, dirs, files in walkList:
		for fileName in files:
			if extensions is None or len(extensions) == 0:
				fileList.append(os.path.abspath(os.path.join(root, fileName)))
			else:
				name, fileExtension = os.path.splitext(fileName)
				if fileExtension in extensions:
					fileList.append(os.path.abspath(os.path.join(root, fileName)))

	return fileList

def extractFilename(filePath):
	"""
	Extract just the file name (no path, no extension) from a filePath
	"""
	return os.path.splitext(os.path.basename(filePath))[0]

def listFiles(dirPath):
	"""
	List all files (not directories) in a directory (does not recurse)
	"""
	if dirPath is None or dirPath == '':
		raise ArgumentException("listFiles requires non-blank dirPath argument")

	return [os.path.abspath(os.path.join(dirPath, fileName)) for fileName in os.listdir(dirPath) if not os.path.isdir(os.path.join(dirPath, fileName))]

def listDirs(dirPath):
	"""
	List all immediate subdirectories of a provided directory (does not recurse)
	"""
	if dirPath is None or dirPath == '':
		raise ArgumentException("listDirs requires non-blank dirPath argument")

	return [os.path.abspath(os.path.join(dirPath, subDirPath)) for subDirPath in os.listdir(dirPath) if os.path.isdir(os.path.join(dirPath, subDirPath))]

