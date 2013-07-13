
from UML import loadTrainingAndTesting
from UML import normalizeData
from UML import crossValidateReturnBest
from UML import functionCombinations


if __name__ == "__main__":

	#### Some sample use cases ####

	###############################################
	#mean normalize your training and testing data#

	trainX, trainY, testX, testY = loadTrainingAndTesting("myFile.txt", "predictionLabels", fractionForTestSet=.15) #load and split up the data
	normalizeData(trainX, testX, algorithm="mean") #perform mean normalization

	##################################################
	#cross validate over some different possibilities#

	#these are the runs we're going to try
	run1 = 'normalizeData(trainX, testX, algorithm="dropFeatures", parameters={"start":1,"end":3}); runAlgorithmWithPeformance(trainX, trainY, testX, testY, algorithm="mlpy.svm")'
	run2 = 'normalizeData(trainX, testX, algorithm="dropFeatures", parameters={"start":1,"end":5}); runAlgorithmWithPeformance(trainX, trainY, testX, testY, algorithm="mlpy.svm")'

	#this will return the text of whichever function performed better, as well as that best performance value
	bestFunction, performance = crossValidateReturnBest(trainX, trainY,'min', [run1, run2], numFolds=10)

	#####################################################
	#cross validate over a large number of possibilities#

	#we'll be trying all combinations of C in [0.1, 1, 10, 100] and iterations in [100, 1000]
	runs = functionCombinations('runAlgorithmWithPeformance(trainX, trainY, testX, testY, algorithm="mlpy.svm", parameters={"C":<0.1|1|10|100>, "iterations":<100|1000>})')

	#this will return the text of whichever function performed better, as well as that best performance value
	bestFunction, performance = crossValidateReturnBest(trainX, trainY, 'min', runs, numFolds=10)





