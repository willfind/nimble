import nltk
import types
from collections import Counter
from scipy import *
from scipy.sparse import *
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.stem.lancaster import LancasterStemmer

import UML
from UML.uml_loading.data_loading import dirMapper
from UML.utility import ArgumentException, EmptyFileException
from UML import data


def readDirectoryIntoSparseMatrix(directoryPath, extensions=['.txt', '.html'], featureMergeMode='all', tokenizingArgDict=None):
	"""
	Convert text files in directoryPath into a sparse matrix, with one document per row.  Also calculates
	# of documents each feature occurs in, and provides mappings from feature -> column index
	and column index -> feature.  Performs pre-processing of text in the files - tokenizing, stop 
	word removal, etc.  (See docstring for convertToTokens() for available processing parameters.)

	Parameters:
		directoryPath: path to a folder containing files to be processed.  
	"""
	# set up data structures to hold aggregate data - map of all features and their indices,
	# number of features encountered so far, number of rows added so far, and a map counting 
	# number of documents each feature has occurred in.
	featureIndexMap = {}
	inverseFeatureIndexMap = {}
	featureCount = 0
	rowCount = 0
	docTermCount = {}
	idRowIndexMap = {}

	#get map of typeName->filePathList
	if featureMergeMode == 'all':
		fileMap = dirMapper(directoryPath, extensions, 'oneType')
	elif featureMergeMode == 'multiTyped':
		fileMap = dirMapper(directoryPath, extensions, 'multiTyped')
	else:
		raise ArgumentException("Unrecognized featureMerge mode in readDirectoryIntoFreqMaps")

	# sparse matrix to hold results.  To start, has as many rows as there are document ids
	# and 10,000,000 columns (we do not expect to use that many columns but there is no penalty
	# for having unused columns.)  
	holderMatrix = dok_matrix((len(fileMap),1000000), dtype='uint16')

	# load all files and convert to token lists/frequency counts, then add to a 
	# 3-layer map: dataType->{docKey->freqMap}
	if featureMergeMode == 'all':
		for docId in fileMap.keys():
			idRowIndexMap[docId] = rowCount
			filePathList = fileMap[docId]['all']
			typeFreqMap = Counter()
			for filePath in filePathList:
				try:
					tokens, freqMap = loadAndTokenize(filePath, tokenizingArgDict)
				except EmptyFileException:
					continue
				freqMap = Counter(freqMap)
				typeFreqMap = typeFreqMap + freqMap
			for token in typeFreqMap.keys():
				if token in featureIndexMap:
					docTermCount[token] = docTermCount[token] + 1
				else:
					featureIndexMap[token] = featureCount
					inverseFeatureIndexMap[featureCount] = token
					docTermCount[token] = 1
					featureCount += 1
				if typeFreqMap[token] >= 65536:
					holderMatrix = holderMatrix.astype('uint32')
				holderMatrix[rowCount, featureIndexMap[token]] = typeFreqMap[token]
			rowCount += 1
	elif featureMergeMode == 'multiTyped':
		for docId in fileMap.keys():
			idRowIndexMap[docId] = rowCount
			typeMap = fileMap[docId]
			for dataType in typeMap.keys():
				filePathList = typeMap[dataType]
				typeFreqMap = Counter()
				for filePath in filePathList:
					try:
						tokens, freqMap = loadAndTokenize(filePath, tokenizingArgDict)
					except EmptyFileException:
						continue
					newFreqMap = Counter()
					for token, count in freqMap.iteritems():
						newFreqMap[dataType + "_" + token] = count
					typeFreqMap = typeFreqMap + newFreqMap
				for token in typeFreqMap.keys():
					if token in featureIndexMap:
						docTermCount[token] = docTermCount[token] + 1
					else:
						featureIndexMap[token] = featureCount
						inverseFeatureIndexMap[featureCount] = token
						docTermCount[token] = 1
						featureCount += 1
					if typeFreqMap[token] >= 65536:
						holderMatrix = holderMatrix.astype('uint32')
					holderMatrix[rowCount, featureIndexMap[token]] = typeFreqMap[token]
			rowCount += 1

	holderMatrix.resize((len(fileMap), featureCount))

	return featureIndexMap, inverseFeatureIndexMap, docTermCount, idRowIndexMap, holderMatrix


def loadAndTokenize(textFilePath, customTokenizingArgs=None):
	if not isinstance(textFilePath, str):
		raise ArgumentException("Invalid argument to defaultLoadAndTokenize: "+repr(textFilePath))

	textFile = open(textFilePath, 'rU')

	text = textFile.read()

	if text is None or text == '':
		raise EmptyFileException("Attempted to tokenize empty file in loadAndTokenize")

	if customTokenizingArgs is None:
		tokens, frequencyMap = convertToTokens(text)
	else:
		customTokenizingArgs['text'] = text
		tokens, frequencyMap = convertToTokens(**customTokenizingArgs)

	return tokens, frequencyMap


def convertToTokens(text, cleanHtml=True, ignoreCase=True, tokenizer='default', removeBlankTokens=True, skipSymbolSet=UML.defaultSkipSetNonAlphaNumeric, removeTokensContaining=None, keepNumbers=False, stopWordSet=UML.defaultStopWords, tokenTransformFunction=None, stemmer='default'):
	"""
		Convert raw text to processed tokens.  Returns a list of tokens in the same order in
		which they appear in the original text, as well as a map of token->frequencyCount.  
		The only required input is the text to be processed.  If the user wants to use their
		own versions of any of the various default processing tasks, they can be passed as
		arguments.

		Inputs:

			text - raw text to be tokenized, as a string

			tokenizer - either a function object which can take a string of raw text and return
			a list of string tokens, or a string identifying a tokenizer which can be instantiated
			and used.  Default tokenizer is the NLTK default english tokenizer.

			removeBlankTokens - set to True if blank tokens should be removed from the results (immediately
			after tokenizing, but before subsequent processing); False if they should be included in results.
			Only applies to blank tokens created by stemming - blank tokens created by deleting symbols or
			other filtering will still be removed.

			skipSymbolSet - a set (or list) of character symbols which should be removed from tokens
			after they have been tokenized, but before any stemming or other transformations have been
			performed.  Example:  if skipSymbolList contains @, then the token f@t would be converted
			to ft.  By default, includes all non-alpha characters.  

			removeTokensContaining - a set (or list) of character symbols, which, if any token contains
			any of them, will result in that token being removed from the results.  Blank by default.

			keepNumbers - if true, retain numeric tokens/characters in the results.  Otherwise, remove
			them from tokens (treat as if they are in skipSymbolList)

			transformer - function that will be run after tokenizing, symbol filtering, and stop word
			removal but before stemming.  Can be a function object or a string identifying a transformer
			module. Must take one token and return a transformed token (string).
			If the returned token is the blank string and removeBlankTokens is False, that token
			will be removed from the results.  Otherwise, it will be included.  By default, no transformation
			function is run.

			stemmer - function that will be run last, presumably a stemmer though any function that
			takes a string and returns a string will work.  Can be a function object or a string 
			representing a Stemming module with a stem(string) function.  By default, uses the NLTK
			Lancaster stemmer.

			TODO: update dosctring to be in accordance with function signature
	"""
	if text is None or text == '':
		raise ArgumentException("convertToTokens requires a non-empty input string")

	if cleanHtml:
		text = nltk.clean_html(text)

	if tokenizer == 'default':
		tokenizer = PunktWordTokenizer()
		tokens = tokenizer.tokenize(text)
	elif isinstance(tokenizer, types.FunctionType):
		tokens = tokenizer(text)
	elif isinstance(tokenizer, str):
		#If tokenizer is a string other than 'default', we assume that it's the name
		# of a tokenizer in nltk's tokenize package
		if tokenizer.startswith("nltk.tokenize"):
			moduleName = tokenizer.rpartition('.')[0]
		else:
			moduleName = "nltk.tokenize"+tokenizer.rpartition('.')[0]
		className = tokenizer.rpartition('.')[2]
		exec("from "+moduleName+" import "+className)
		exec("tokenizerObj = "+className+"()")
		tokens = tokenizerObj.tokenize(text)
	else:
		raise ArgumentException("convertToTokens requires a tokenizer")

	if ignoreCase:
		lowerCaseTokens = []
		for token in tokens:
			lowerCaseTokens.append(token.lower())
		tokens = lowerCaseTokens

	#set the collection of stopwords to be used, if stop word filtering is turned on
	if stopWordSet is not None:
		#Set the set of stop words: read in a file, use a passed list, or use the default list
		if isinstance(stopWordSet, str):
			stopWordSet = loadStopList(stopWordSet)
		elif isinstance(stopWordSet, (list, set)):
			pass
		else:
			raise ArgumentException("Unrecognized stop list type in convertToTokens: "+type(stopWordSet))

		#filter out tokens that are in the set of stop words
		tokens = filterStopWords(tokens, stopWordSet)

	#Remove all tokens containing any symbol in the set removeTokensContaining
	if removeTokensContaining is not None:
		flagSymbolFilteredSet = []
		for token in tokens:
			removeToken = False
			for character in token:
				if character in removeTokensContaining:
					removeToken = True
			if not removeToken:
				flagSymbolFilteredSet.append(token)
		tokens = flagSymbolFilteredSet

	if skipSymbolSet is not None:
		filteredSet = []
		for token in tokens:
			if token == '':
				if not removeBlankTokens:
					filteredSet.append(token)
			else:
				filteredToken = token.translate(None, skipSymbolSet)
				if filteredToken != '':
					filteredSet.append(filteredToken)
		tokens = filteredSet

	if not keepNumbers:
		filteredSet = []
		numbersSet = UML.numericalChars
		for token in tokens:
			if token == '' and not removeBlankTokens:
				filteredSet.append(token)
			else:
				filteredToken = token.translate(None, numbersSet)
				if filteredToken != '':
					filteredSet.append(filteredToken)
		tokens = filteredSet

	# May need to run stop word filtering again, as there may be 
	# more matches after symbol removal
	if stopWordSet is not None and skipSymbolSet is not None:
		tokens = filterStopWords(tokens, stopWordSet)

	if tokenTransformFunction is not None:
		transformedSet = []
		for token in tokens:
			transformedSet.append(tokenTransformFunction(token))
		tokens = transformedSet

	if stemmer == 'default':
		stemmer = LancasterStemmer()
		stemmedTokens = []
		for token in tokens:
			stemmedTokens.append(stemmer.stem(token))
		tokens = stemmedTokens

	frequencyMap = convertTokenListToFrequencyMap(tokens)

	return tokens, frequencyMap

def filterStopWords(tokens, stopWords=set()):
	"""
	Provided a list of tokens and a set of stop words, return a list of tokens
	from which stop words have been removed. Original order is maintained, 
	minus any tokens that have been removed.
	"""
	if tokens is None:
		raise ArgumentException("List of tokens in filterStopWords must not be None")

	filteredList = []
	for token in tokens:
		if token not in stopWords:
			filteredList.append(token)

	return filteredList

def convertToBaseData(dokMatrix, idRowIndexMap, featureDocCount, featureColumnIndexMap, columnIndexFeatureMap, representationMode='count', labelIndexList=[0]):
	"""
	Provided a set of data objects containing all necessary information about a corpus/data set,
	convert to a CSC BaseData object with the desired feature representation (options are binary, 
	term frequency, and TF-IDF).
	TODO: complete docstring
	"""
	if dokMatrix is None:
		raise ArgumentException("dokMatrix cannot be null")
	elif idRowIndexMap is None:
		raise ArgumentException("idRowIndexMap cannot be null")
	elif featureDocCount is None:
		raise ArgumentException("featureDocCount cannot be null")
	elif featureColumnIndexMap is None:
		raise ArgumentException("featureColumnIndexMap cannot be null")
	elif columnIndexFeatureMap is None:
		raise ArgumentException("columnIndexFeatureMap cannot be null")


	if representationMode.lower() == 'binary':
		newDokMatrix = dok_matrix(dokMatrix.shape, dokMatrix.dtype)
		numRows = dokMatrix.shape[0]
		numColumns = dokMatrix.shape[1]
		for i in range(numRows):
			for j in range(numColumns):
				if j in labelIndexList:
					newDokMatrix[i, j] = dokMatrix[i, j]
				if dokMatrix[i, j] > 0:
					newDokMatrix[i, j] = 1
		dokMatrix = newDokMatrix
	elif representationMode.lower() == 'frequency':
		pass
	elif representationMode.lower() == 'tfidf':
		newDokMatrix = dok_matrix(dokMatrix.shape, dokMatrix.dtype)
		numRows = dokMatrix.shape[0]
		numColumns = dokMatrix.shape[1]
		idfMap = calcIdfValues(featureDocCount, numRows)
		for i in range(numRows):
			tfIdfMap = calcTfIdfVals(dokMatrix, i, columnIndexFeatureMap, idfMap, labelIndexList)
			for j in range(numColumns):
				if dokMatrix[i, j] > 0:
					tfIdfVal = tfIdfMap[j]
					newDokMatrix[i, j] = tfIdfVal
	else:
		raise ArgumentException("Unrecognized representationMode: " + str(representationMode))

	cooVersion = newDokMatrix.tocoo()

	featureNameList = [''] * len(featureColumnIndexMap)
	for feature, columnIndex in featureColumnIndexMap.iteritems():
		featureNameList[columnIndex] = feature

	baseDataVersion = data('coo', cooVersion, featureNameList, sendToLog=False)

			
def calcTfIdfVals(dokMatrix, rowIndex, columnIndexFeatureMap, idfMap, labelIndexlist):
	"""
	Calculate the tf normalization factor for one document (row) in
	a dok matrix with feature frequency counts in each entry.  Returns a map
	of columnIndex:tfIdf value for the specified row.  Ignores 0-valued entries
	in dokMatrix, so the final map only contains mappings for columns that have
	non-zero values.
	"""
	weightSum = 0.0
	columnIndexTfIdfTempMap = {}
	finalTfIdfMap = {}
	for j in range(dokMatrix.shape[1]):
		if j in labelIndexList:
			columnIndexTfIdfTempMap[j] = dokMatrix[rowIndex, j]
		feature = columnIndexFeatureMap[j]
		if dokMatrix[rowIndex, j] > 0:
			tempTfIdf = (dokMatrix[rowIndex, j] + 1) * idfMap[feature]
			columnIndexTfIdfTempMap[j] = tempTfIdf
			weightSum += tempTfIdf**2
	normalizationFactor = math.sqrt(weightSum)
	for j in range(dokMatrix.shape[1]):
		finalTfIdfMap[j] = columnIndexTfIdfTempMap[j] / normalizationFactor

	return finalTfIdfMap


def calcIdfValues(featureDocCountMap, numDocs):
	"""
	Calculate the Inverse Document Frequency (IDF) for each feature
	in featureDocCountMap.  featureDocCountMap is a dictionary associating
	each feature in a corpus with the number of documents in which it appears.
	numDocs is the total number of documents in the corpus.  Returns a dictionary
	associating feature to its IDF, as a float.
	"""
	log2 = math.log(2)

	featureIdfMap = {}
	for feature, count in featureDocCountMap:
		featureIdfMap[feature] = math.log((float(numDocs) / float(count)) / log2

	return featureIdfMap

def convertAttributeMapToMatrix(attributeIdMap, attributeName='default', attributeTransformFunction=None):
	"""
	Convert a map of {documentId:attribute} into a BaseData
	object.  Assumes that all attributes in the map are of one type, and
	that each document can have no more than one of any given type of attribute.
	"""
	idRowIndexMap = {}
	attributeColumnIndexMap = {}
	columnIndexAttributeMap = {}
	attributeDocCount = {}
	rowCount = 0
	uniqueAttributeCount = 0

	holderMatrix = dok_matrix((len(attributeIdMap), len(attributeIdMap)), dtype='uint8')

	for docId, attribute in attributeIdMap.iteritems():
		if attributeTransformFunction is not None:
			attribute = attributeTransformFunction(attribute)
		attribute = attributeName + "_" + attribute
		if attribute in attributeColumnIndexMap:
			idRowIndexMap[docId] = rowCount
			holderMatrix[rowCount, attributeColumnIndexMap[attribute]] = 1
			attributeDocCount[attribute] += 1
		else:
			attributeColumnIndexMap[attribute] = uniqueAttributeCount
			columnIndexAttributeMap[uniqueAttributeCount] = attribute
			idRowIndexMap[docId] = rowCount
			attributeDocCount[attribute] = 1
			uniqueAttributeCount += 1
			holderMatrix[rowCount, attributeColumnIndexMap[attribute]] = 1
		rowCount += 1

	holderMatrix.resize((rowCount + 1, uniqueAttributeCount + 1))
	return idRowIndexMap, attributeDocCount, attributeColumnIndexMap, columnIndexAttributeMap, holderMatrix


def convertTextToTokenList(text, textProcessingFunc=None):
	"""
	Given a string containing text, convert to a list of tokens, using Python's 
	nltk: tokenize, remove odd symbols, remove stop words, stem.
	"""
	if not isinstance(text, str):
		raise ArgumentException

	#if a processingFunc was passed, we pass the raw text to that function
	#and return the results
	if textProcessingFunc is not None:
		return textProcessingFunc(text)

	#remove html and tokenize
	text = nltk.clean_html(text)
	tokens = nltk.word_tokenize(text)

	#create list of undesirable chars to be filtered, and get list of stop words
	delchars = ''.join(c for c in map(chr, range(256)) if not c.isalpha())
	stopWords = loadStopList()

	#filter out numbers, unwanted symbols, blank strings, and stop words
	filteredTokens = []
	for token in tokens:
		filteredToken = token.translate(None, delchars).lower()
		if filteredToken != '' and filteredToken not in stopWords:
			filteredTokens.append(filteredToken)

	stemmer = LancasterStemmer()
	processedTokens = []
	for filteredToken in filteredTokens:
		processedToken = stemmer.stem(filteredToken)
		processedTokens.append(processedToken)

	return processedTokens

def joinDokMatrixDatSet(dokSetList, idLabelMap=None):
	"""
	Join together a list of data sets.  Each data set must be contained
	in a dictionary containing the following data objects.  Assumes that
	there is no overlap in the features of any of the data sets.  There can
	be overlap between document ids of different data sets.

		featureDocCountMap:  a map of features to the # of docs in the data
		set in which they appear

		featureColumnIndexMap: a map of feature to the index of the column
		representing the feature in the dok matrix.

		columnIndexFeatureMap: a map of column index to name/content of the
		feature represented in that column

		docIdRowIndexMap: a map of document Id to the row in the dok matrix
		containing that document's feature data

		dokMatrix: a dok matrix, with each row representing one document
		and each column representing one 
	"""
	numUniqueDocuments = 0
	featureCount = 0
	featureDocCountMapAll = {}
	featureColumnIndexMapAll = {}
	columnIndexFeatureNameMapAll = {}
	docIdRowIndexMapAll = {}
	
	approxDocCount = 0
	approxFeatureCount = 0
	maxDtype = 'uint16'
	for dataSetMap in dokSetList:
		approxDocCount += len(dataSetMap['docIdRowIndexMap'])
		approxFeatureCount += len(dataSetMap['featureColumnIndexMap'])
		if dataSetMap['dokMatrix'].dtype == 'uint32':
			maxDtype = 'uint32'

	if idLabelMap is not None:
		approxFeatureCount += 1
		featureCount += 1
		featureColumnIndexMapAll['classLabel'] = 1
		columnIndexFeatureNameMapAll[0] = 'classLabel'

	dataMatrixAll = dok_matrix((approxDocCount, approxFeatureCount), dtype=maxDtype)

	for dataSetMap in dokSetList:
		featureDocCountMap = dataSetMap['featureDocCountMap']
		featureColumnIndexMap = dataSetMap['featureColumnIndexMap']
		columnIndexFeatureMap = dataSetMap['columnIndexFeatureMap']
		docIdRowIndexMap = dataSetMap['docIdRowIndexMap']
		dataMatrix = dataSetMap['dokMatrix']
		matrixWidth = dataMatrix.shape[1]
		for feature, columnIndex in featureColumnIndexMap.iteritems():
			if feature in featureColumnIndexMapAll:
				if feature in featureDocCountMapAll:
					featureDocCountMapAll[feature] += featureDocCountMap[feature]
				else:
					featureDocCountMapAll[feature] = featureDocCountMap[feature]
			elif feature not in featureColumnIndexMapAll:
				featureColumnIndexMapAll[feature] = featureCount
				columnIndexFeatureNameMapAll[featureCount] = feature
				featureCount += 1
				if feature in featureDocCountMapAll:
					featureDocCountMapAll[feature] += featureDocCountMap[feature]
				else:
					featureDocCountMapAll[feature] = featureDocCountMap[feature]
		for docId, rowIndex in docIdRowIndexMap:
			if docId in docIdRowIndexMapAll:
				newRowIndex = docIdRowIndexMapAll[docId]
			else:
				newRowIndex = numUniqueDocuments
				docIdRowIndexMapAll[docId] = numUniqueDocuments
				numUniqueDocuments += 1
			if idLabelMap is not None:
				label = idLabelMap[docId]
				dataMatrixAll[newRowIndex, 0] = label
			for j in range(matrixWidth):
				if dataMatrix[rowIndex, j] > 0:
					feature = columnIndexFeatureMap[j]
					newColumnIndex = columnIndexFeatureNameMapAll[feature]
					dataMatrixAll[newRowIndex, newColumnIndex] = dataMatrix[rowIndex, j]

	dataMatrixAll.resize((numUniqueDocuments, featureCount))

	return docIdRowIndexMapAll, featureDocCountMapAll, featureColumnIndexMapAll, columnIndexFeatureNameMapAll, dataMatrixAll



def convertTokenListToFrequencyMap(tokenList):
	"""
	Given a string containing text, convert to a map of 
	processed tokens->tokenFrequency, as a collections.Counter object
	TODO: complete docstring
	"""
	if tokenList is None or len(tokenList) == 0:
		return None

	#Convert text to tokens using convertTextToTokenList, possibly using
	#user-defined function
	tokenCountMap = Counter()
	for token in tokenList:
		if token in tokenCountMap:
			tokenCountMap[token] = tokenCountMap[token] + 1
		else:
			tokenCountMap[token] = 1

	return tokenCountMap

def computeCorpusTermCountMap(docTermCountsList):
	"""
	Provided a list of dictionaries, where each dictionary maps tokens
	to frequency counts and represents one document, return a dictionary
	associating each token in the corpus with the number of documents in which
	that token appears.
	"""
	corpusDocumentTermFrequencies = {}
	for doc in docTermCountsList:
		for term, count in doc.iteritems():
			if term in corpusDocumentTermFrequencies and count > 0:
				corpusDocumentTermFrequencies[term] += 1
			else:
				corpusDocumentTermFrequencies[term] = 1

	return corpusDocumentTermFrequencies

def loadStopList(stopListFilePath=None, ignoreCase=True):
	"""
	Load a stop list from the specified file.  If no file is specified, load
	the default nltk stop word list.  If a filePath is passed, assumes that
	the file has one stop word per line.
	"""
	if stopListFilePath is None or not isinstance(stopListFilePath, str):
		raise ArgumentException("stopListFilePath must be a string "
			+"pointing to the location of a file containing stop words")

	stopListFile = open(stopListFilePath, 'rU')
	stopList = []
	for line in stopListFile:
		if line != '':
			if ignoreCase:
				stopList.append(line.lower().strip())
			else:
				stopList.append(line.strip())

	return stopList

