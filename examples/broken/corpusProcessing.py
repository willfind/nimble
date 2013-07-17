"""
	Example of how to convert a set of text files on the disk into a matrix market file/BaseData
	object.  Includes loading, processing, and merging/join operations.
"""

from allowImports import boilerplate
boilerplate()

import os.path
import UML

from UML.read.data_loading import *
from UML.read.text_processing import *

if __name__ == "__main__":
	#convert a directory w/text files (or *.html files) into a set of data objects
	testDirPath = os.path.join(UML.UMLPath, 'datasets/testDirectory')

	#read in the directory of individual files and convert to a dok sparse matrix + auxiliary
	# data objects
	featureIndexMap, inverseFeatureIndexMap, docTermCount, idRowIndexMap, holderMatrix = readDirectoryIntoSparseMatrix(testDirPath, featureMergeMode="multiTyped")

	#manually create a map of docId:attribute (one attribute per document)
	attributeDocDict = {'001':'dog', '002':'cat', '003':'rat', '004':'cat', '005':'dog', '006':'cat', '007':'cat', '008':'rat', '009':'horse'}
	
	#manually create a map of docId:class Label (one per document)
	labelDict = {'001':1, '002': 2, '003':2, '004':2, '005':1, '006':2, '007':1, '008':1, '009':1}

	#Convert attribute map to 2-dimensional matrix + auxiliary data objects, same format as text files
	idRowIndexMapAttr, docTermCountAttr, featureIndexMapAttr, inverseFeatureIndexMapAttr, holderMatrixAttr = convertAttributeMapToMatrix(attributeDocDict)

	textDataDict = {'featureDocCountMap':docTermCount, 
					'featureColumnIndexMap':featureIndexMap, 
					'columnIndexFeatureMap':inverseFeatureIndexMap,
					'docIdRowIndexMap':idRowIndexMap,
					'dokMatrix':holderMatrix}

	attributeDataDict = {'featureDocCountMap':docTermCountAttr, 
						'featureColumnIndexMap':featureIndexMapAttr, 
						'columnIndexFeatureMap':inverseFeatureIndexMapAttr,
						'docIdRowIndexMap':idRowIndexMapAttr,
						'dokMatrix':holderMatrixAttr}

	dataSetList = [textDataDict, attributeDataDict]

	#Join the attribute data set to the text data set using docId as the key
	featureIndexMapFinal, inverseFeatureIndexMapFinal, docTermCountFinal, idRowIndexMapFinal, holderMatrixFinal = joinDokMatrixDataSet(dataSetList, labelDict)

	#Convert data sets (dok matrix + auxiliary data objects) to a COO Base Data object
	cooVersion = convertToBaseData(holderMatrixFinal, idRowIndexMapFinal, docTermCountFinal, featureIndexMapFinal, inverseFeatureIndexMapFinal, representationMode='tfidf')

