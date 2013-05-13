"""
	
"""
import math
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
		Empty constructor
		"""
		self.isEmpty = True
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
		if featureMergeMode == 'all':
			fileMap = dirMapper(directoryPath, extensions, 'oneType')
		elif featureMergeMode == 'multiTyped':
			fileMap = dirMapper(directoryPath, extensions, 'multiTyped')
		else:
			raise ArgumentException("Unrecognized featureMerge mode in readDirectoryIntoFreqMaps")

		# sparse matrix to hold results.  To start, has as many rows as there are document ids
		# and 10,000,000 columns (we do not expect to use that many columns but there is no penalty
		# for having unused columns.)  
		self.data = dok_matrix((len(fileMap),1000000), dtype='uint16')

		# load all files and convert to token lists/frequency counts, then add to a 
		# 3-layer map: dataType->{docKey->freqMap}
		for docId in fileMap.keys():
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
					newFreqMap = Counter()
					for token, count in freqMap.iteritems():
						if dataType == 'all':
							newFreqMap["allText_" + token] = count
						else:
							newFreqMap[dataType + "_" + token] = count
					typeFreqMap = typeFreqMap + newFreqMap
				for token in typeFreqMap.keys():
					if token in self.featureColumnIndexMap:
						self.featureDocCountMap[token] = self.featureDocCountMap[token] + 1
					else:
						self.featureColumnIndexMap[token] = featureCount
						self.columnIndexFeatureMap[featureCount] = token
						self.featureDocCountMap[token] = 1
						featureCount += 1
					if typeFreqMap[token] >= 65535:
						self.data = self.data.astype('uint32')
					self.data[rowCount, self.featureColumnIndexMap[token]] = typeFreqMap[token]
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
			if attribute in self.featureColumnIndexMap:
				self.docIdRowIndexMap[docId] = rowCount
				self.rowIndexDocIdMap[rowCount] = docId
				self.data[rowCount, self.featureColumnIndexMap[attribute]] = 1
				self.featureDocCountMap[attribute] += 1
			else:
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
		Merge the provided DokDataSet object into this object.  Generally assumes that there is
		either an overlap between the documents in toMerge and this object, or the features,
		or neither, but not both.  If there is overlap between both features and document
		IDs, token/feature frequency counts will be added together.  If this
		object is empty, it will hold a copy of the data in toMerge.
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
		for listIndex in range(len(nonZeroEntries[0])):
			rowIndex = nonZeroEntries[0][listIndex]
			columnIndex = nonZeroEntries[1][listIndex]
			dataMatrixAll[rowIndex, columnIndex] = self.data[rowIndex, columnIndex]

		#use these two numbers to track the index of new features or documents as
		#they are added to dataMatrixAll
		numUniqueDocumentsCounter = len(self.docIdRowIndexMap)
		numFeaturesCounter = len(self.featureColumnIndexMap)

		#Combine feature doc counts for the two data sets
		self.featureDocCountMap += toMerge.featureDocCountMap

		#add all features in new DokDataSet to current feature map
		for feature in toMerge.featureColumnIndexMap.keys():
			if feature not in self.featureColumnIndexMap:
				self.featureColumnIndexMap[feature] = numFeaturesCounter
				self.columnIndexFeatureMap[numFeaturesCounter] = feature
				numFeaturesCounter += 1

		entriesToMerge = toMerge.data.nonzero()
		for listIndex in range(len(entriesToMerge[0])):
			oldRowIndex = entriesToMerge[0][listIndex]
			oldColumnIndex = entriesToMerge[1][listIndex]
			docId = toMerge.rowIndexDocIdMap[oldRowIndex]
			feature = toMerge.columnIndexFeatureMap[oldColumnIndex]
			if docId in self.docIdRowIndexMap:
				newRowIndex = self.docIdRowIndexMap[docId]
			else:
				newRowIndex = numUniqueDocumentsCounter
				self.docIdRowIndexMap[docId] = numUniqueDocumentsCounter
				self.rowIndexDocIdMap[numUniqueDocumentsCounter] = docId
				numUniqueDocumentsCounter += 1
			newColumnIndex = self.featureColumnIndexMap[feature]
			if dataMatrixAll[newRowIndex, newColumnIndex] > 0:
				dataMatrixAll[newRowIndex, newColumnIndex] += toMerge.data[oldRowIndex, oldColumnIndex]
			else:
				dataMatrixAll[newRowIndex, newColumnIndex] = toMerge.data[oldRowIndex, oldColumnIndex]


		dataMatrixAll.resize((numUniqueDocumentsCounter, numFeaturesCounter))
		self.data = dataMatrixAll

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


	def toCooBaseData(self, representationMode='frequency'):
		"""
		Provided a set of data objects containing all necessary information about a corpus/data set,
		convert this object to a COO BaseData object with the desired feature representation (options are binary, 
		term frequency, and TF-IDF).
		The resulting object will have column headers based on features contained in this object.  Document IDs
		and class labels, if present, should be in the leftmost columns of the final object.  
		"""
		if self.isEmpty:
			raise ImproperActionException("Can't convert empty object to BaseData version")

		if representationMode.lower() == 'binary':
			newDokMatrix = dok_matrix(self.data.shape, 'uint8')
			entries = self.data.nonzero()
			for l in range(len(entries[0])):
				rowIndex = entries[0][l]
				columnIndex = entries[1][l]
				newDokMatrix[rowIndex, columnIndex] = 1
		elif representationMode.lower() == 'frequency':
			newDokMatrix = self.data
		elif representationMode.lower() == 'tfidf':
			newDokMatrix = dok_matrix(self.data.shape, dtype='float32')
			numRows = self.data.shape[0]
			self.calcIdfValues()
			for i in range(numRows):
				tfIdfMap = self.calcTfIdfVals(i)
				for columnIndex, tfIdfVal in tfIdfMap.iteritems():
					newDokMatrix[i, columnIndex] = tfIdfVal
		else:
			raise ArgumentException("Unrecognized representationMode: " + str(representationMode))

		cooVersion = newDokMatrix.tocoo()

		featureNameList = [''] * len(self.featureColumnIndexMap)
		for feature, columnIndex in self.featureColumnIndexMap.iteritems():
			featureNameList[columnIndex] = feature

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
				if docId in self.docIdRowIndexMap:
					rowIndex = self.docIdRowIndexMap[docId]
				else:
					continue
				labelDokMatrix[rowIndex, labelColumnIndex] = classLabel
			idLabelOrderedNames.append(classLabelName+ '***')
			labelColumnIndex += 1
		#convert dok matrix w/labels and ids to Coo BaseData version
		labelsAndIds = data('coo', labelDokMatrix.tocoo(), idLabelOrderedNames, sendToLog=False)

		#put together the two matrices, with doc Ids and class labels in leftmost columns
		labelsAndIds.appendFeatures(baseDataVersion)

		return labelsAndIds


	def calcTfIdfVals(self, rowIndex):
		"""
		Calculate the tf normalization factor for one document (row) in
		a dok matrix with feature frequency counts in each entry.  Returns a map
		of columnIndex:tfIdf value for the specified row.  Ignores 0-valued entries
		in dokMatrix, so the final map only contains mappings for columns that have
		non-zero values.
		"""
		if self.featureIdfMap is None or len(self.featureIdfMap) == 0:
			self.calcIdfValues()

		weightSum = 0.0
		columnIndexTfIdfTempMap = {}
		finalTfIdfMap = {}
		for j in range(self.data.shape[1]):
			feature = self.columnIndexFeatureMap[j]
			if self.data[rowIndex, j] > 0:
				tempTfIdf = (self.data[rowIndex, j] + 1) * self.featureIdfMap[feature]
				columnIndexTfIdfTempMap[j] = tempTfIdf
				weightSum += tempTfIdf**2
		normalizationFactor = math.sqrt(weightSum)
		for k in range(self.data.shape[1]):
			if self.data[rowIndex, k] > 0:
				finalTfIdfMap[k] = columnIndexTfIdfTempMap[k] / normalizationFactor

		return finalTfIdfMap


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
