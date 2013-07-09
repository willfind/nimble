"""
Script loading a portion of the adult income data set, and processing the string valued
data into something easier to learn on.

"""

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":

	from UML import create

	# string manipulation to get and make paths
	pathOrig = "../datasets/adult_income_classification_tiny.csv"
	pathOut = "../datasets/adult_income_classification_tiny_numerical.csv"

	# we specify that we want a Matrix object returned, and with just the path it will
	# decide automaticallly the format of the file that is being loaded
	processed = create("Matrix", pathOrig)

	# this feature is a precalculated similarity rating. Lets not make it too easy....
	processed.extractFeatures('fnlwgt')

	#convert assorted features from strings to binary category columns
	processed.featureToBinaryCategoryFeatures('sex')
	processed.featureToBinaryCategoryFeatures('marital-status')
	processed.featureToBinaryCategoryFeatures('occupation')
	processed.featureToBinaryCategoryFeatures('relationship')
	processed.featureToBinaryCategoryFeatures('race')
	processed.featureToBinaryCategoryFeatures('native-country')
	
	# convert 'income' column (the classification label) to a single numerical column
	processed.featureToIntegerCategories('income')

	#scrub the rest of the string valued data -- the ones we converted are the non-redundent ones
	processed.dropStringValuedFeatures()

	# output the split and normalized sets for later usage
	processed.writeFile('csv', pathOut, includeFeatureNames=True)
