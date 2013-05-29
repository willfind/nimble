"""
	
"""
import math
import numpy as np
from numpy import searchsorted
from bisect import bisect_left
from collections import Counter
from scipy.sparse import dok_matrix

import UML
from UML.uml_loading.data_loading import dirMapper
from UML.uml_loading.text_processing import loadAndTokenize
from UML import data
from UML.utility import ArgumentException, EmptyFileException, ImproperActionException

class DokDataSet(object):
	"""
	Class to hold intermediate form of a (probably) textual data set.  Used in converting
	raw text files into a Coo BaseData object or saving to disk as one file, in a format
	that can be easily fed into a Classifier.

	Made up of scipy's dok sparse matrix format, as well as dictionaries holding maps of
	feature:columnIndex, columnIndex:feature, docId:rowIndex, and docId:classLabel.  The 
	dok matrix holds data about the features making up each document; assumes that each
	entry in the matrix represents the number of times that feature (column) occurs in
	that document (row).
	"""

	def __init__(self):
		"""
		Empty constructor.  Set up all data structures for this object except the dok matrix.
		"""
		self.isEmpty = True
		self.containsRawText = False
		self.featureColumnIndexMap = {}
		self.columnIndexFeatureMap = {}
		self.docIdRowIndexMap = {}
		self.rowIndexDocIdMap = {}
		self.featureDocCountMap = Counter()
		self.featureIdfMap = {}
		self.classLabelSets = {}

	def loadDirectory(self, directoryPath, extensions=['.txt', '.html'], featureMergeMode='all', cleanHtml=True, ignoreCase=True, tokenizer='default', removeBlankTokens=True, skipSymbolSet=UML.defaultSkipSetNonAlphaNumeric, removeTokensContaining=None, keepNumbers=False, stopWordSet=UML.defaultStopWords, tokenTransformFunction=None, stemmer='default'):
		"""
		Read all files in a directory, assuming they contain raw text, and load them into this object.  
		For explanation of all parameter options, see docstring of convertToCooBaseData() function in
		convert_to_basedata.py.  
		"""
		# If this DokDataSet object already contains a data set, we will load the new data and
		# merge it into the current data.  This assumes that, while there may be overlap in documents
		# or overlap in features between the two data sets, there is not simultaneous overlap between
		# documents and features.  In that case, some data may get overwritten.
		if not self.containsRawText:
			self.containsRawText = True

		if not self.isEmpty:
			tempDokDataSet = DokDataSet()
			tempDokDataSet.loadDirectory(directoryPath, extensions, featureMergeMode, cleanHtml, ignoreCase, tokenizer, removeBlankTokens, skipSymbolSet, removeTokensContaining, keepNumbers, stopWordSet, tokenTransformFunction, stemmer)
			self.merge(tempDokDataSet)
			return

		# set up data structures to hold aggregate data - map of all features and their indices,
		# number of features encountered so far, number of rows added so far, and a map counting 
		# number of documents each feature has occurred in.
		featureCount = 0
		rowCount = 0

		#get map of typeName->filePathList
		if featureMergeMode.lower() == 'all'.lower():
			fileMap = dirMapper(directoryPath, extensions, 'oneType')
		elif featureMergeMode.lower() == 'multiTyped'.lower():
			fileMap = dirMapper(directoryPath, extensions, 'multiTyped')
		else:
			raise ArgumentException("Unrecognized featureMerge mode in readDirectoryIntoFreqMaps")

		numDocsVisited = 0
		# sparse matrix to hold results.  To start, has as many rows as there are document ids
		# and 10,000,000 columns (we do not expect to use that many columns but there is no penalty
		# for having unused columns.)  
		self.data = dok_matrix((len(fileMap),1000000), dtype='uint16')

		# load all files and convert to token lists/frequency counts, then add to a 
		# 3-layer map: dataType->{docKey->freqMap}
		for docId in fileMap.keys():
			numDocsVisited += 1
			#record the unique key of this document
			self.docIdRowIndexMap[docId] = rowCount
			self.rowIndexDocIdMap[rowCount] = docId
			typeMap = fileMap[docId]
			#Iterate through all data types associated with this document
			for dataType in typeMap.keys():
				filePathList = typeMap[dataType]
				#create a new Counter object to track the number of appearances of each feature
				#in the document
				typeFreqMap = Counter()
				#iterate through all files associated with this document
				for filePath in filePathList:
					try:
						tokens, freqMap = loadAndTokenize(filePath, cleanHtml, ignoreCase, tokenizer, removeBlankTokens, skipSymbolSet, removeTokensContaining, keepNumbers, stopWordSet, tokenTransformFunction, stemmer)
					except EmptyFileException:
						continue
					if tokens is None or freqMap is None or len(tokens) == 0 or len(freqMap) == 0:
						continue
					newFreqMap = Counter()
					for token, count in freqMap.iteritems():
						if dataType == 'all':
							newFreqMap["allText_" + token] = count
						else:
							newFreqMap[dataType + "_" + token] = count
					typeFreqMap = typeFreqMap + newFreqMap
				#loop over all tokens in this document and update count of docs this token
				#appears in.  If the token/feature hasn't been seen before, add to this object's
				#tracking data structures
				for token in typeFreqMap.keys():
					try:
						columnIndex = self.featureColumnIndexMap[token]
						self.featureDocCountMap[token] += 1
					except KeyError:
						columnIndex = featureCount
						self.featureColumnIndexMap[token] = columnIndex
						self.columnIndexFeatureMap[featureCount] = token
						self.featureDocCountMap[token] = 1
						featureCount += 1
					if typeFreqMap[token] >= 65535:
						self.data = self.data.astype('uint32')
					self.data[rowCount, columnIndex] = typeFreqMap[token]
			rowCount += 1

		self.data.resize((len(self.docIdRowIndexMap), featureCount))
		self.isEmpty = False

		return None


	def loadAttributeMap(self, attributeIdMap, attributeName='default', attributeTransformFunction=None):
		"""
		Load a map of {documentId:attribute} into this object.  Assumes that all attributes in the 
		map are of one type, and that each document can have no more than one of any given type of attribute.
		Each unique attribute gets one column in this objects dok matrix.  attributeName should be a unique
		identifier for the type of attribute added.  attributeTransformFunction can be any function that
		takes a string and returns a string.
		"""
		# If this DokDataSet object already contains a data set, we will load the new data and
		# merge it into the current data.  This assumes that, while there may be overlap in documents
		# or overlap in features between the two data sets, there is not simultaneous overlap between
		# documents and features.  In that case, some data may get overwritten.
		if not self.isEmpty:
			tempDokDataSet = DokDataSet()
			tempDokDataSet.loadAttributeMap(attributeIdMap, attributeName, attributeTransformFunction)
			self.merge(tempDokDataSet)
			return

		#set up tracking numbers
		rowCount = 0
		uniqueAttributeCount = 0

		self.data = dok_matrix((len(attributeIdMap), len(attributeIdMap)), dtype='uint8')

		for docId, attribute in attributeIdMap.iteritems():
			if attributeTransformFunction is not None:
				attribute = attributeTransformFunction(attribute)
			attribute = attributeName + "_" + attribute
			try:
				self.data[rowCount, self.featureColumnIndexMap[attribute]] = 1
				self.docIdRowIndexMap[docId] = rowCount
				self.rowIndexDocIdMap[rowCount] = docId
				self.featureDocCountMap[attribute] += 1
			except KeyError:
				self.featureColumnIndexMap[attribute] = uniqueAttributeCount
				self.columnIndexFeatureMap[uniqueAttributeCount] = attribute
				self.docIdRowIndexMap[docId] = rowCount
				self.rowIndexDocIdMap[rowCount] = docId
				self.featureDocCountMap[attribute] = 1
				uniqueAttributeCount += 1
				self.data[rowCount, self.featureColumnIndexMap[attribute]] = 1
			rowCount += 1

		self.data.resize((rowCount, uniqueAttributeCount))

		self.isEmpty = False

		return None

	def merge(self, toMerge):
		"""
		Merge the provided DokDataSet object into this object.  Operates like a left join,
		with this object being the left data set and toMerge being the right. If there is overlap
		between both features and document IDs, token/feature frequency counts will be added together.
		If this object is empty, it will become a deep copy of the data in toMerge.
		"""
		if toMerge is None:
			raise ArgumentException("toMerge cannot be empty")
		elif self.isEmpty:
			self.featureColumnIndexMap = toMerge.featureColumnIndexMap.copy()
			self.columnIndexFeatureMap = toMerge.columnIndexFeatureMap.copy()
			self.docIdRowIndexMap = toMerge.docIdRowIndexMap.copy()
			self.featureDocCountMap = toMerge.featureColumnIndexMap.copy()
			self.data = toMerge.data.copy()
			self.isEmpty = False
			return
		
		approxDocCount = len(self.docIdRowIndexMap) + len(toMerge.docIdRowIndexMap)
		approxFeatureCount = len(self.featureColumnIndexMap) + len(toMerge.featureColumnIndexMap)

		maxDtype = self.data.dtype
		if maxDtype == 'uint16' and toMerge.data.dtype == 'uint32':
			maxDtype = 'uint32'
		elif maxDtype != 'uint32' and toMerge.data.dtype == 'uint64':
			maxDtype = 'uint64'

		dataMatrixAll = dok_matrix((approxDocCount, approxFeatureCount), dtype=maxDtype)

		nonZeroEntries = self.data.nonzero()

		#populate the new matrix with all entries from self.data
		for listIndex in xrange(len(nonZeroEntries[0])):
			rowIndex = nonZeroEntries[0][listIndex]
			columnIndex = nonZeroEntries[1][listIndex]
			dataMatrixAll[rowIndex, columnIndex] = self.data[rowIndex, columnIndex]

		#use these two numbers to track the index of new features or documents as
		#they are added to dataMatrixAll
		numUniqueDocumentsCounter = len(self.docIdRowIndexMap)
		numFeaturesCounter = len(self.featureColumnIndexMap)

		#merge all entries from toMerge into this data set.  Update auxiliary data sets
		#as we go
		entriesToMerge = toMerge.data.nonzero()
		for listIndex in xrange(len(entriesToMerge[0])):
			oldRowIndex = entriesToMerge[0][listIndex]
			oldColumnIndex = entriesToMerge[1][listIndex]
			docId = toMerge.rowIndexDocIdMap[oldRowIndex]
			feature = toMerge.columnIndexFeatureMap[oldColumnIndex]

			#if the new docId is already present in this data set, we
			#add new data to docId's row in this object.  Otherwise,
			#if toMerge contains raw text data, we add the new document
			#to this object.  If toMerge only contains attribute data,
			#we ignore the new document (we don't want documents that have
			#no text data).
			try:
				newRowIndex = self.docIdRowIndexMap[docId]
			except KeyError:
				if toMerge.containsRawText:
					newRowIndex = numUniqueDocumentsCounter
					numUniqueDocumentsCounter += 1
					self.rowIndexDocIdMap[newRowIndex] = docId
					self.docIdRowIndexMap[docId] = newRowIndex
				else:
					continue

			#if this feature is already in feature map, get its index.
			#Otherwise, add it to the feature map
			try:
				newColumnIndex = self.featureColumnIndexMap[feature]
			except KeyError:
				self.featureColumnIndexMap[feature] = numFeaturesCounter
				self.columnIndexFeatureMap[numFeaturesCounter] = feature
				newColumnIndex = self.featureColumnIndexMap[feature]
				numFeaturesCounter += 1

			if dataMatrixAll[newRowIndex, newColumnIndex] > 0:
				dataMatrixAll[newRowIndex, newColumnIndex] += toMerge.data[oldRowIndex, oldColumnIndex]
			else:
				dataMatrixAll[newRowIndex, newColumnIndex] = toMerge.data[oldRowIndex, oldColumnIndex]


		dataMatrixAll.resize((numUniqueDocumentsCounter, numFeaturesCounter))
		self.data = dataMatrixAll

		self.reCalcFeatureDocCount()

		return None

	def addClassLabelMap(self, docIdClassLabelMap, classLabelName=None):
		"""
		Add a column of class labels to the beginning of self.data (at column 0).
		Shifts all other columns to the right by one.  If classLabelName is None,
		uses the default name of 'classLabel' post-pended with the number of columns containing
		class labels (so if this is the second column of class labels, the default
		name will be 'classLabel2').

		Class Labels should be represented numerically.
		"""
		if len(self.classLabelSets) > 1 and classLabelName is None:
			raise ImproperActionException("Please provide a name/type for your class label list")
		elif classLabelName in self.classLabelSets:
			raise ImproperActionException("That classLabelName is already in use; please choose another")
		else:
			if classLabelName is None:
				self.classLabelSets['defaultClassLabel'] = docIdClassLabelMap
			else:
				self.classLabelSets[classLabelName] = docIdClassLabelMap


	def toCooBaseData(self, representationMode='frequency', minTermFrequency=2, featureTypeWeightScheme=None):
		"""
		Provided a set of data objects containing all necessary information about a corpus/data set,
		convert this object to a COO BaseData object with the desired feature representation (options are binary,
		term frequency, and TF-IDF).
		The resulting object will have column headers based on features contained in this object.  Document IDs
		and class labels, if present, should be in the leftmost columns of the final object.

		Inputs:
			representationMode: Can be one of three strings representing standard types of feature
			                    representation: 'binary', 'frequency' (for term frequency), or
			                    'tfidf'.  Alternately, can be a function which will be called for
			                    each feature in each document, and will compute a value
			                    for feature representation based on the following inputs:

			                    	feature: the feature/term itself, as a string

			                    	featureCount:  Absolute number of times this feature appears in 
			                    	this document (row).  Integer.
			                    	
			                    	numDocs:  Number of documents in the data set. Integer

			                    	numFeatures: Number of features in the data set. Integer
			                    	
			                    	featureDocCount: Number of documents in the corpus in which the
			                    	feature/term appears. Integer.

			                    	docFeatureCount: Number of features in the current document (current row). Integer.

			                    	docId: unique ID of the current document. Integer.

			minTermFrequency: Minimum number of documents a feature/term must appear in to be kept in the
							  corpus.  If any feature doesn't occur in at least this many documents, it will
							  be removed from the corpus entirely.


		"""
		if self.isEmpty:
			raise ImproperActionException("Can't convert empty object to BaseData version")

		featureNameList = self.removeInfrequentFeatures(minTermFrequency)

		#If any of the featureTypeWeights is not an integer, we need to change the type of
		#this objects data matrix to float
		if featureTypeWeightScheme is not None and len(featureTypeWeightScheme) > 0:
			for featureTypeName, featureTypeWeight in featureTypeWeightScheme.iteritems():
				if not isinstance(featureTypeWeight, int):
					self.data = self.data.asfptype()
					break

		if featureTypeWeightScheme is not None and representationMode.lower() != 'binary':
			entries = self.data.nonzero()
			for l in xrange(len(entries[0])):
				rowIndex = entries[0][l]
				columnIndex = entries[1][l]
				feature = self.columnIndexFeatureMap[columnIndex]
				featureType = feature.partition('_')[0]
				if featureType in featureTypeWeightScheme:
					self.data[rowIndex, columnIndex] *= featureTypeWeightScheme[featureType]

		if isinstance(representationMode, str):
			if representationMode.lower() == 'binary':
				newDokMatrix = dok_matrix(self.data.shape, 'uint8')
				entries = self.data.nonzero()
				for l in xrange(len(entries[0])):
					rowIndex = entries[0][l]
					columnIndex = entries[1][l]
					newDokMatrix[rowIndex, columnIndex] = 1
			elif representationMode.lower() == 'frequency':
				#this object's data matrix is already in the format we want
				newDokMatrix = self.data
			elif representationMode.lower() == 'tfidf':
				newDokMatrix = self.calcTfIdfVals()

		elif hasattr(representationMode, '__call__'):
			newDokMatrix = dok_matrix(self.data.shape, 'float64')
			entries = self.data.nonzero()
			for l in xrange(len(entries[0])):
				rowIndex = entries[0][l]
				columnIndex = entries[1][l]
				numNonZeroInRow = self.data.getrow(rowIndex).getnnz()
				feature = self.columnIndexFeatureMap[columnIndex]
				newDokMatrix[rowIndex, columnIndex] = representationMode(feature, self.data[rowIndex, columnIndex], self.data.shape[0], len(featureNameList), self.featureDocCountMap[feature], numNonZeroInRow, int(self.rowIndexDocIdMap[rowIndex]))

		cooVersion = newDokMatrix.tocoo()

		baseDataVersion = data('coo', cooVersion, featureNameList, sendToLog=False)

		# Build a dok matrix containing document Ids and Class Labels
		labelDokMatrix = dok_matrix((newDokMatrix.shape[0], 1 + len(self.classLabelSets)))
		idLabelOrderedNames = []
		for docId, rowIndex in self.docIdRowIndexMap.iteritems():
			labelDokMatrix[rowIndex, 0] = int(docId)
		idLabelOrderedNames.append('documentId***')
		labelColumnIndex = 1
		for classLabelName, classLabelSet in self.classLabelSets.iteritems():
			for docId, classLabel in classLabelSet.iteritems():
				try:
					rowIndex = self.docIdRowIndexMap[docId]
				except KeyError:
					continue
				labelDokMatrix[rowIndex, labelColumnIndex] = classLabel
			idLabelOrderedNames.append(classLabelName+ '***')
			labelColumnIndex += 1
		#convert dok matrix w/labels and ids to Coo BaseData version
		labelsAndIds = data('coo', labelDokMatrix.tocoo(), idLabelOrderedNames, sendToLog=False)

		#put together the two matrices, with doc Ids and class labels in leftmost columns
		labelsAndIds.appendFeatures(baseDataVersion)

		return labelsAndIds


	def calcTfIdfVals(self):
		"""
		Calculate the tf normalization factor for one document (row) in
		a dok matrix with feature frequency counts in each entry.  Returns a map
		of columnIndex:tfIdf value for the specified row.  Ignores 0-valued entries
		in dokMatrix, so the final map only contains mappings for columns that have
		non-zero values.
		"""
		if self.featureIdfMap is None or len(self.featureIdfMap) == 0:
			self.calcIdfValues()

		rowIndexWeightSumMap = {}
		nonZeroEntries = self.data.nonzero()

		tfIdfDataHolder = dok_matrix(self.data.shape, 'float64')

		for i in xrange(len(nonZeroEntries[0])):
			rowIndex = nonZeroEntries[0][i]
			columnIndex = nonZeroEntries[1][i]
			feature = self.columnIndexFeatureMap[columnIndex]
			featureCount = self.data[rowIndex, columnIndex]
			tempTfIdf = (featureCount + 1) * self.featureIdfMap[feature]
			tfIdfDataHolder[rowIndex, columnIndex] = tempTfIdf
			try:
				rowIndexWeightSumMap[rowIndex] += tempTfIdf**2
			except KeyError:
				rowIndexWeightSumMap[rowIndex] = tempTfIdf**2
				

		for rowIndex, normalizationFactor in rowIndexWeightSumMap.iteritems():
			rowIndexWeightSumMap[rowIndex] = math.sqrt(normalizationFactor)

		for i in xrange(len(nonZeroEntries[0])):
			rowIndex = nonZeroEntries[0][i]
			columnIndex = nonZeroEntries[1][i]
			normalizationFactor = rowIndexWeightSumMap[rowIndex]
			tempTfIdf = tfIdfDataHolder[rowIndex, columnIndex]
			tfIdfDataHolder[rowIndex, columnIndex] = tempTfIdf / normalizationFactor

		return tfIdfDataHolder


	def calcIdfValues(self):
		"""
		Calculate the Inverse Document Frequency (IDF) for each feature
		in featureDocCountMap.  featureDocCountMap is a dictionary associating
		each feature in a corpus with the number of documents in which it appears.
		numDocs is the total number of documents in the corpus.  Returns a dictionary
		associating feature to its IDF, as a float.
		"""
		log2 = math.log(2)

		numDocs = self.data.shape[0]
		for feature, count in self.featureDocCountMap.iteritems():
			self.featureIdfMap[feature] = math.log((float(numDocs) / float(count))) / log2

		return None

	def reCalcFeatureDocCount(self):
		"""
			Re-calculate the count of # of documents each feature appears in,
			stored in self.featureDocCountMap
		"""
		featureDocCountMap = {}
		nonZeroEntries = self.data.nonzero()
		for i in xrange(len(nonZeroEntries[0])):
			columnIndex = nonZeroEntries[1][i]
			feature = self.columnIndexFeatureMap[columnIndex]
			try:
				featureDocCountMap[feature] += 1
			except KeyError:
				featureDocCountMap[feature] = 1

		self.featureDocCountMap = featureDocCountMap

	def removeInfrequentFeatures(self, minFeatureFrequency):
		"""
		remove all features that don't appear in at least minFeatureFrequency
		documents, and adjust all data objects/maps to reflect the changes.
		"""

		#set up data objects to track changes: new list of feature names, set of
		#features that do not occur frequently enough to retain, and a list of the
		#column indices of features that will be removed
		featureNameList = [''] * len(self.featureColumnIndexMap)
		featuresToRemove = set()
		columnShiftIndexList = []

		#go through all current features and find those that need to be removed
		for feature, columnIndex in self.featureColumnIndexMap.iteritems():
			featureDocCount = self.featureDocCountMap[feature]
			if featureDocCount >= minFeatureFrequency:
				featureNameList[columnIndex] = feature
			else:
				featuresToRemove.add(feature)
				columnShiftIndexList.append(columnIndex)
				del self.featureDocCountMap[feature]

		#Use columnShiftIndexList to figure out how much each retained feature
		#needs to have its column index shifted by
		columnShiftIndexList.sort()
		columnsToRemoveSet = set(columnShiftIndexList)
		columnShiftIndexList = np.array(columnShiftIndexList, np.int32)
		totalColumnShift = len(columnShiftIndexList)
		newRowCount = self.data.shape[0]
		newColumnCount = self.data.shape[1] - totalColumnShift

		while True:
			try:
				featureNameList.remove('')
			except ValueError:
				break

		#set up new data structures: we will reassign this objects matrix and columnIndex
		#maps to these new structures
		newDokMatrix = dok_matrix((newRowCount, newColumnCount), self.data.dtype)
		newColumnIndexFeatureMap = {}
		newFeatureColumnIndexMap = {}
		entries = self.data.nonzero()
		for l in xrange(len(entries[0])):
			rowIndex = entries[0][l]
			columnIndex = entries[1][l]
			
			if columnIndex not in columnsToRemoveSet:
				newColumnIndex = columnIndex - searchsorted(columnShiftIndexList, columnIndex)
				feature = self.columnIndexFeatureMap[columnIndex]
				newColumnIndexFeatureMap[newColumnIndex] = feature
				newFeatureColumnIndexMap[feature] = newColumnIndex
				newDokMatrix[rowIndex, newColumnIndex] = self.data[rowIndex, columnIndex]

		self.columnIndexFeatureMap = newColumnIndexFeatureMap
		self.featureColumnIndexMap = newFeatureColumnIndexMap
		self.data = newDokMatrix

		#return the list of feature names.  The index of each feature name is its column
		#index in self.data
		return featureNameList




def calcFirstBiggerIndex(columnIndexShiftList, searchItem):
	"""
	Provided a list of column indices, sorted in ascending order,
	find the index of the first entry in that list that is larger
	than or equal to searchIndex.  If searchIndex is larger than all
	items in the list, return len(columnIndexShiftMap)

	Example:  if the list is [3, 6, 14, 25], and searchIndex is
	10, this function will return 2 (which is the index of 14 in the list).
	If searchIndex is 27, this function will return 4.
	"""
	for i in xrange(len(columnIndexShiftList)):
		currItem = columnIndexShiftList[i]
		if searchItem <= currItem:
			return i

	return len(columnIndexShiftList)






