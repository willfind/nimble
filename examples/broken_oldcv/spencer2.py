

from allowImports import boilerplate
boilerplate()
import UML
import os.path

if __name__ == "__main__":

	from UML import crossValidateReturnBest
	from UML import functionCombinations
	from UML import createData
	from UML import runAndTest
	from UML.metrics import fractionIncorrect


#	print "UML dir", dir(UML)
	# path to input specified by command line argument
	pathIn = os.path.join(UML.UMLPath, "datasets/sparseSample.mtx")
	allData = createData('sparse', pathIn, fileType="mtx")

	print "data loaded"

	yData = allData.extractFeatures([5])
	xData = allData

	yData = yData.copyAs(format="Matrix")

	print "data formatted"

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRun = 'runAndTest("shogun.MulticlassOCAS", trainX, trainY, testX, testY, {"C":<1.0>}, [fractionIncorrect])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTest':runAndTest, 'fractionIncorrect':fractionIncorrect}

	print "runs prepared"

	bestFunction, performance = crossValidateReturnBest(xData, yData, runs, mode='min', numFolds=5, extraParams=extraParams)
	print bestFunction
	print performance

