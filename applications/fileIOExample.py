"""
Script loading some data, spliting it, and writing the results to seperate files

"""

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":

	from UML import data

	# string manipulation to get and make paths
	pathOrig = "example_data/adult_income_classification_tiny.csv"
	pathSplit = pathOrig.rsplit('.',1)
	if len(pathSplit) > 1:
		suffix = '.' + pathSplit[1]
	else:
		suffix = ""
	pathTrain = pathSplit[0] + "TRAIN" + suffix
	pathTest = pathSplit[0] + "TEST" + suffix

	# we specify that we want a RowListData object returned, and with just the path it will
	# decide automaticallly the format of the file that is being loaded
	train = data("RowListData", pathOrig)

	#scrub the set of any string valued data
	train.dropStringValuedFeatures()

	# split off a random test set
	total = train.points()
	num = int(round(.15*total))
	test = train.extractPoints(start=0, end=total, number=num, randomize=True)

	# output the split and normalized sets for later usage
	train.writeFile('csv', pathTrain, includeFeatureNames=False)
	test.writeFile('csv', pathTest, includeFeatureNames=False)

