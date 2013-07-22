
from allowImports import boilerplate
boilerplate()

import datetime

if __name__ == "__main__":

	
	import os.path

	import UML

	from UML import orderedCrossValidateReturnBest
	from UML import functionCombinations
	from UML import runAndTest
	from UML import createData
	from UML.metrics import rootMeanSquareError

	# load orig apple data
	appleStock = createData("List", data=os.path.join(UML.UMLPath, "datasets/appleStock.csv"))

	# convert date string to days from start
	dateStrings = appleStock.extractFeatures("Date")
	def makeDate(point):
		tokens = point[0].split("-")
		return datetime.date(int(tokens[0]),int(tokens[1]),int(tokens[2]))
	dateObjects = dateStrings.applyToEachPoint(makeDate)
	dateObjects.setFeatureName(0, "Date")
	appleStock.appendFeatures(dateObjects)
	appleStock.sortPoints("Date")
	dateNum = appleStock.featureNames["Date"]
	start = appleStock[0,"Date"]
	def makeDifference(point):
		return (point[dateNum] - start).days
	daysSinceStart = appleStock.applyToEachPoint(makeDifference)
	daysSinceStart.setFeatureName(0, "Time")
	appleStock.extractFeatures("Date")
	appleStock.appendFeatures(daysSinceStart)

	# add prediction column
	adjCloseNextDay = appleStock.copyFeatures("Adj Close")
	adjCloseNextDay.setFeatureName(0, "Adj Close Next Day")
	adjCloseNextDay.extractPoints(0)
	appleStock.extractPoints(appleStock.points()-1)
	appleStock.appendFeatures(adjCloseNextDay)
	adjCloseNum = appleStock.featureNames["Adj Close"]
	adjCloseNDNum = appleStock.featureNames["Adj Close Next Day"]
	def ratioChange(point):
		return (float(point[adjCloseNDNum]) / point[adjCloseNum]) - 1
	predictionLabels = appleStock.applyToEachPoint(ratioChange)
	predictionLabels.setFeatureName(0, "Predict")
	appleStock.extractFeatures(["Adj Close Next Day"])

	appleStock = appleStock.toMatrix()

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRun = 'runAndTest("sciKitLearn.LinearRegression", trainX, trainY, testX, testY, {}, [rootMeanSquareError])["rootMeanSquareError"]'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTest':runAndTest, 'rootMeanSquareError':rootMeanSquareError}

	bestFunction, performance = orderedCrossValidateReturnBest(appleStock, predictionLabels, runs, mode='min', orderedFeature="Time", minTrainSize=datetime.timedelta(days=365), maxTrainSize=datetime.timedelta(days=730), stepSize=datetime.timedelta(days=90), gap=0, minTestSize=1, maxTestSize=1, extraParams=extraParams)
	print bestFunction
	print performance


