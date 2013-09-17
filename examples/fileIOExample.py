"""
Script loading some data, spliting it, and writing the results to seperate files

"""

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":
	import os.path
	import UML
	from UML import createData

	# string manipulation to get and make paths
	pathOrig = os.path.join(UML.UMLPath, "datasets/adult_income_classification_tiny.csv")
	pathSplit = pathOrig.rsplit('.',1)
	if len(pathSplit) > 1:
		suffix = '.' + pathSplit[1]
	else:
		suffix = ""
	pathTrain = pathSplit[0] + "TRAIN" + suffix
	pathTest = pathSplit[0] + "TEST" + suffix

	# we specify that we want a List object returned, and with just the path it will
	# decide automaticallly the format of the file that is being loaded
	train = createData("List", pathOrig)

	#scrub the set of any string valued data
	train.dropFeaturesContainingType(basestring)

	# split off a random test set
	total = train.pointCount
	num = int(round(.15*total))
	test = train.extractPoints(start=0, end=total, number=num, randomize=True)

	# output the split and normalized sets for later usage
	train.writeFile(pathTrain, format='csv', includeFeatureNames=True)
	test.writeFile(pathTest, format='csv', includeFeatureNames=True)

