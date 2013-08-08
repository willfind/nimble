"""
Contains various error metrics to be used for evaluating the results
of supervised learning algorithms.


"""
import inspect
import UML
from math import sqrt
from UML.exceptions import ArgumentException
from UML.umlHelpers import computeError



def rootMeanSquareError(knownValues, predictedValues):
	"""
		Compute the root mean square error.  Assumes that knownValues and predictedValues contain
		numerical values, rather than categorical data.
	"""
	return computeError(knownValues, predictedValues, lambda x,y,z: z + (y - x)**2, lambda x,y: sqrt(x/y))

def meanAbsoluteError(knownValues, predictedValues):
	"""
		Compute mean absolute error. Assumes that knownValues and predictedValues contain
		numerical values, rather than categorical data.
	"""
	return computeError(knownValues, predictedValues, lambda x,y,z: z + abs(y - x), lambda x,y: x/y)

def fractionIncorrect(knownValues, predictedValues):
	"""
		Compute the proportion of incorrect predictions within a set of
		instances.  Assumes that values in knownValues and predictedValues are categorical.
	"""
	return computeError(knownValues, predictedValues, lambda x,y,z: z if x == y else z + 1, lambda x,y: x/y)

def fractionTrueNegativeTop90(knownValues, predictedValues, negativeLabel):
	"""
		Wrapper function for fractionTrueNegative.  Computes the proportion
		of posts that fall in the 90% most likely to be in the positive class
		that are actually in the negative class.  Assumes there are only 2 classes
		in the data set, and that predictedValues contains a score for each of
		the 2 labels.  Sorts by (positive label score - negative label score), in
		ascending order, and looks at highest 90% of values.
				
	"""
	return fractionTrueNegative(knownValues, predictedValues, negativeLabel, proportionToScore=0.90)

def fractionTrueNegativeTop50(knownValues, predictedValues, negativeLabel):
	"""
		Wrapper function for fractionTrueNegative.  Computes the proportion
		of posts that fall in the 50% most likely to be in the positive class
		that are actually in the negative class.  Assumes there are only 2 classes
		in the data set, and that predictedValues contains a score for each of
		the 2 labels.  Sorts by (positive label score - negative label score), in
		ascending order, and looks at highest 50% of values.
	"""
	return fractionTrueNegative(knownValues, predictedValues, negativeLabel, proportionToScore=0.50)

def fractionTrueNegativeBottom10(knownValues, predictedValues, negativeLabel):
	"""
		Wrapper function for fractionTrueNegative.  Computes the proportion
		of posts that fall in the 50% most likely to be in the positive class
		that are actually in the negative class.  Assumes there are only 2 classes
		in the data set, and that predictedValues contains a score for each of
		the 2 labels.  Sorts by (positive label score - negative label score), in
		ascending order, and looks at lowest 10% of values.
	"""
	return fractionTrueNegative(knownValues, predictedValues, negativeLabel, proportionToScore=0.10, reverseSort=True)

def fractionTrueNegative(knownValues, labelScoreList, negativeLabel, proportionToScore=0.90, reverseSort=False):
	"""
		Computes the proportion
		of posts that fall in the x% most likely to be in the positive class
		that are actually in the negative class.  Assumes there are only 2 classes
		in the data set, and that predictedValues contains a score for each of
		the 2 labels.  Sorts by (positive label score - negative label score), in
		ascending order, and looks at highest x proportion of values, where x is defined
		by proportionToScore.  If reverseSort is True, looks at lowest x proportion of values.
	"""
	#proportion must fall between 0 and 1
	if proportionToScore <= 0.0 or proportionToScore > 1.0:
		raise ArgumentException("proportionToScore must be between 0 and 1.0")

	#use featureNames in labelScoreList to discover what the positiveLabel is
	labelNames = labelScoreList.featureNames
	positiveLabel = ''

	#Enforce requirement that there be only 2 classes present
	if len(labelNames) != 2:
		raise ArgumentException("fractionTrueNegative requires a set of precisely two predicted label scores for each point")

	#look through featureNames; whichever isn't the negative label must be
	#the positive label
	for labelName in labelNames.keys():
		if labelName == negativeLabel:
			continue
		else:
			positiveLabel = labelName
			break

	negativeLabelIndex = labelScoreList.featureNames[negativeLabel]
	positiveLabelIndex = labelScoreList.featureNames[positiveLabel]

	#Compute the score difference (positive label score - negative label score) for
	#all entries in labelScoreList
	scoreDifferenceList = []
	labelScoreList = labelScoreList.copy(asType="python list")
	for i in range(len(labelScoreList)):
		positiveScore = labelScoreList[i][positiveLabelIndex]
		negativeScore = labelScoreList[i][negativeLabelIndex]
		scoreDiff = positiveScore - negativeScore
		scoreDifferenceList.append(scoreDiff)

	#convert knownValues to list of lists
	listOfKnownLabels = knownValues.copy(asType="python list")
	knownLabels = listOfKnownLabels[0:]
	for i in range(len(knownLabels)):
		knownLabels[i] = knownLabels[i][0]

	#Put together score differences and known labels, then sort by score difference,
	#so we have a list ranked by likelihood of having positiveLabel.  Generally will
	#be in descending order, so we can look at those points that are most likely to 
	#be positive.  
	scoreDiffAndKnown = zip(scoreDifferenceList, knownLabels)

	if reverseSort is True:
		scoreDiffAndKnown.sort(key=lambda score: score[0])
	else:
		scoreDiffAndKnown.sort(key=lambda score: score[0], reverse=True)

	#Get some proportion of list based on proportionToScore
	topProportionIndex = int(round(proportionToScore * len(scoreDiffAndKnown)))
	sortedTopProportion = scoreDiffAndKnown[0:topProportionIndex]

	#Unzip into two lists
	sortedScoreDiffAndKnown = ([scoreDiff for scoreDiff,known in sortedTopProportion], [known for scoreDiff,known in sortedTopProportion])

	#get newly sorted known labels
	sortedKnownValues = sortedScoreDiffAndKnown[1]

	#compute number of negative labels present in specified proportion of posts that
	#are predicted to be more likely to be positive labels
	numNegativeLabels = 0
	for knownLabel in sortedKnownValues:
		if str(knownLabel) == negativeLabel:
			numNegativeLabels += 1

	#return proportion of top posts that are negative
	return float(numNegativeLabels) / float(len(sortedKnownValues))

def fractionIncorrectBottom10(knownValues, labelScoreList, negativeLabel):
	"""
		Note: this error function is only appropriate for binary classification
		situations.  If there are more than two labels in the labelScoreMap,
		it will break.
		Compute the proportion of incorrect predictions in the bottom 10% of
		predictions.  Bottom 10% is defined by sorting all predictions by
		the following metric: positiveLabelScore - negativeLabelScore, then
		computing the classification error only for those points whose metric
		fall within the lowest proportionToScore of points.
	"""
	#figure out the positive label
	labelNames = labelScoreList.featureNames
	positiveLabel = ''
	if len(labelNames) != 2:
		raise ArgumentException("fractionTrueNegative requires a set of precisely two predicted label scores for each point")

	for labelName in labelNames.keys():
		if labelName == negativeLabel:
			continue
		else:
			positiveLabel = labelName

	negativeLabelIndex = labelScoreList.featureNames[negativeLabel]
	positiveLabelIndex = labelScoreList.featureNames[positiveLabel]

	#Compute the score difference (positive label score - negative label score) for
	#all entries in labelScoreList
	scoreDifferenceList = []
	labelScoreList = labelScoreList.copy(asType="python list")
	for i in range(len(labelScoreList)):
		positiveScore = labelScoreList[i][positiveLabelIndex]
		negativeScore = labelScoreList[i][negativeLabelIndex]
		scoreDiff = positiveScore - negativeScore
		scoreDifferenceList.append(scoreDiff)

	#convert knownValues to list of lists; drop first row, which has featureNames
	listOfKnownLabels = knownValues.copy(asType="python list")
	knownLabels = listOfKnownLabels[0:][0]

	#Put together score differences and known labels, then sort by score difference,
	#so we have a list ranked, in descending order, by most likely to have negative label
	scoreDiffAndKnown = zip(scoreDifferenceList, knownLabels)

	scoreDiffAndKnown.sort(key=lambda score: score[0])

	#Get bottom of list (lowest score differences, which may be negative)
	topProportionIndex = int(round(0.10 * len(scoreDiffAndKnown)))
	sortedTopProportion = scoreDiffAndKnown[0:]

	#Unzip into two lists
	sortedScoreDiffAndKnown = ([scoreDiff for scoreDiff,known in sortedTopProportion], [known for scoreDiff,known in sortedTopProportion])

	#get newly sorted known labels
	sortedKnownValues = sortedScoreDiffAndKnown[1]

	#compute number of negative labels present in specified proportion of posts that
	#are predicted to be more likely to be positive labels
	winningLabels = []
	for scoreDiff in sortedScoreDiffAndKnown[0]:
		if scoreDiff <= 0.0:
			winningLabels.append(negativeLabel)
		else:
			winningLabels.append(positiveLabel)

	correctPredictions = 0
	for i in range(len(winningLabels)):
		if sortedKnownValues[i] == winningLabels[i]:
			correctPredictions += 1

	proportionCorrect = float(correctPredictions) / float(len(sortedKnownValues))

	return proportionCorrect



def detectBestResult(functionToCheck):
	"""
	Provides sample data to the function in question and evaluates the results
	to determine whether the returned value associates correctness with
	minimum values or maximum values.

	functionToCheck may only take two or three arguments. In the two argument
	case, the first must be a vector of desired values and the second must be
	a vector of predicted values. In the second case, the first argument must
	be a vector of known labes, the second argument must be an object
	containing confidence scores for different labels, and the third argument
	must be the value of a label value present in the data. In either cause,
	the functions must return a float value.

	"""
	(args, varargs, keywords, defaults) = inspect.getargspec(functionToCheck)
	# we are in the known / predicted parameter case
	if len(args) == 2:
		knownRaw = [[0],[1],[0],[1],[0]]
		correctRaw = [[0],[1],[0],[1],[0]]
		wrongRaw = [[1],[0],[1],[0],[1]]

		known = UML.createData(retType="List", data=knownRaw)
		correct = UML.createData(retType="List", data=correctRaw)
		wrong = UML.createData(retType="List", data=wrongRaw)

		correctScore = functionToCheck(known, correct)
		wrongScore = functionToCheck(known, wrong)

		if correctScore > wrongScore:
			return "max"
		elif correctScore < wrongScore:
			return 'min'
		else:
			raise ArgumentException("Unable to differentiate best result for input funciton")
	elif len(args) == 3:
		knownRaw = [[1],[1],[0],[1],[0],[1],[0],[1],[0],[0]]
		correctRaw = [[-1,1],[-.99,.99],[.95,-.95],[-.88,.88],[.85,-.85],[-.77,.77],[.75,-.75],[-.66,.66],[.65,-.65],[.55,-.55]]
		wrongRaw = [[1,-1],[.99,-.99],[-.95,.95],[.88,-.88],[-.85,.85],[.77,-.77],[-.75,.75],[.66,-.66],[-.65,.65],[-.55,.55]]
		
		known = UML.createData(retType="List", data=knownRaw)
		correct = UML.createData(retType="List", data=correctRaw, featureNames=['0','1'])
		wrong = UML.createData(retType="List", data=wrongRaw, featureNames=['0','1'])

		correctScore = functionToCheck(known, correct, 0)
		wrongScore = functionToCheck(known, wrong, 0)

		print correctScore
		print wrongScore
#		assert False

		if correctScore > wrongScore:
			return "max"
		elif correctScore < wrongScore:
			return 'min'
		else:
			raise ArgumentException("Unable to differentiate best result for input funciton")
	else:
		raise ArgumentException("function takes wrong number of parameters, unable to do detection")











