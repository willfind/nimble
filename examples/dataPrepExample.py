"""
Demonstrates loading and processing some string valued data into a dataset that
is purely numerical. Uses a portion of the UCI ML repository census income data
set (aka Adult).

"""

from __future__ import absolute_import
try:
    from allowImports import boilerplate
except:
    from .allowImports import boilerplate
import six

boilerplate()

if __name__ == "__main__":
    import os.path
    import UML

    from UML import createData

    # string manipulation to get and make paths
    pathOrig = os.path.join(UML.UMLPath, "datasets/adult_income_classification_tiny.csv")
    pathOut = os.path.join(UML.UMLPath, "datasets/adult_income_classification_tiny_numerical.csv")

    # we specify that we want a List object returned, and that we want the first row to
    # taken as the featureNames. Given the .csv extension on the path it will infer the
    # format of the file to be loaded, though that could be explicitly specified if desired.
    processed = createData("List", pathOrig, featureNames=True, useLog=True)

    # this feature is a precalculated similarity rating. Let's not make it too easy...
    processed.extractFeatures('fnlwgt')

    # convert assorted features from strings to binary category columns
    processed.replaceFeatureWithBinaryFeatures('sex')
    processed.replaceFeatureWithBinaryFeatures('marital-status')
    processed.replaceFeatureWithBinaryFeatures('occupation')
    processed.replaceFeatureWithBinaryFeatures('relationship')
    processed.replaceFeatureWithBinaryFeatures('race')
    processed.replaceFeatureWithBinaryFeatures('native-country')

    # convert 'income' column (the classification label) to a numerical labels as opposed to
    # string named classes.
    processed.transformFeatureToIntegers('income')

    # scrub the rest of the string valued data -- the ones we converted are the non-redundant ones
    processed.dropFeaturesContainingType(six.string_types)

    # output the cleaned data set for later usage
    processed.writeFile(pathOut, includeNames=True)
