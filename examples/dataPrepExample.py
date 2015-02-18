"""
Script loading a portion of the adult income data set, and processing the string valued
data into something easier to learn on.

"""

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":
	import os.path
	import UML

	from UML import createData

	# string manipulation to get and make paths
	pathOrig = os.path.join(UML.UMLPath, "datasets/adult_income_classification_tiny.csv")
	pathOut = os.path.join(UML.UMLPath, "datasets/adult_income_classification_tiny_numerical.csv")

	# we specify that we want a List object returned, and with just the path it will
	# decide automatically the format of the file that is being loaded
	processed = createData("List", pathOrig)

	# this feature is a precalculated similarity rating. Lets not make it too easy....
	processed.extractFeatures('fnlwgt')

	#convert assorted features from strings to binary category columns
	processed.replaceFeatureWithBinaryFeatures('sex')
	processed.replaceFeatureWithBinaryFeatures('marital-status')
	processed.replaceFeatureWithBinaryFeatures('occupation')
	processed.replaceFeatureWithBinaryFeatures('relationship')
	processed.replaceFeatureWithBinaryFeatures('race')
	processed.replaceFeatureWithBinaryFeatures('native-country')
	
	# convert 'income' column (the classification label) to a single numerical column
	processed.transformFeatureToIntegers('income')

	#scrub the rest of the string valued data -- the ones we converted are the non-redundant ones
	processed.dropFeaturesContainingType(basestring)

	# output the split and normalized sets for later usage
	processed.writeFile(pathOut, includeNames=True)
