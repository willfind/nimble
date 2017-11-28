"""
Demonstrates loading, intelligently spliting, and then writing data to seperate
files. Uses a portion of the UCI ML repository census income data set (aka Adult).

"""

from allowImports import boilerplate

boilerplate()

if __name__ == "__main__":
    import os.path
    import UML
    from UML import createData

    # string manipulation to get and make paths
    pathOrig = os.path.join(UML.UMLPath, "datasets/adult_income_classification_tiny.csv")
    pathSplit = pathOrig.rsplit('.')
    pathTrain = pathSplit[0] + "TRAIN" + ".csv"
    pathTest = pathSplit[0] + "TEST" + ".csv"

    # we specify that we want a Matrix object returned, and that we want the first row to
    # taken as the featureNames. Given the .csv extension on the path it will infer the
    # format of the file to be loaded, though that could be explicitly specified if desired.
    full = createData("DataFrame", pathOrig, featureNames=True)

    # scrub the set of any string valued data
    full.dropFeaturesContainingType(basestring)

    # demonstrate splitting the data in train and test sets out of place. By default,
    # the distribution of points between the two returned objects is random.
    testFraction = .15
    (trainOutPlace, testOutPlace) = full.trainAndTestSets(testFraction)

    # demonstrate splitting the data into train and test sets in place
    total = full.points
    num = int(round(testFraction * total))
    testInPlace = full.extractPoints(start=0, end=total-1, number=num, randomize=True)
    trainInPlace = full

    # the two methods yield comparable results
    assert testInPlace.points == num
    assert trainInPlace.points == total - num
    assert testInPlace.points == testOutPlace.points
    assert trainInPlace.points == trainOutPlace.points

    # output the split and normalized sets for later usage
    trainInPlace.writeFile(pathTrain, format='csv', includeNames=True)
    testInPlace.writeFile(pathTest, format='csv', includeNames=True)
