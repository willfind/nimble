import pdb
import sys
import numpy
import scipy
import os.path
import math
import copy
from functools import partial

import matplotlib.pyplot as plt

KDE_HELPER_PATH = "/home/tpburns/Dropbox/ML_intern_tpb/python_workspace/kdePlotting"
ORIG_HELPER_PATH = "/home/tpburns/Dropbox/ML_intern_tpb/python_workspace/"
sys.path.append(KDE_HELPER_PATH)
sys.path.append(ORIG_HELPER_PATH)
from fancyHistogram import kdeAndHistogramPlot
from multiplot import plotDualWithBlendFill

from allowImports import boilerplate
boilerplate()

import UML
from UML.randomness import pythonRandom
from UML.randomness import numpyRandom


def scotts_factor(pCount, fCount):
	n = pCount
	d = fCount
	return numpy.power(n, -1./(d+4))

def silver_factor(pCount, fCount):
	n = pCount
	d = fCount
	return numpy.power(n*(d+2.0)/4.0, -1./(d+4))


def makePowerOfAdjustedLogLikelihoodSum(exp):
	def PowerOfAdjustedLogLikelihoodSum(knownValues, predictedValues):
		ratioValues = predictedValues / predictedValues.pointStatistics('max').featureStatistics('max')[0]  # everything now on scale from 0 to 1
		negativeLogOfRatios = (-numpy.log(ratioValues))
		expNegLog = negativeLogOfRatios ** exp
		total = numpy.sum(expNegLog)

		return total

	PowerOfAdjustedLogLikelihoodSum.optimal = 'min'	

	return PowerOfAdjustedLogLikelihoodSum



def LogLikelihoodSum(knownValues, predictedValues):
	"""
	Where predictedValues contains the likelihood of each point given the
	model we are testing, take the log of each likelihood and sum the
	results.

	"""
#	print predictedValues.data.shape
	negLog = -numpy.log2(predictedValues.copyAs("numpyarray"))
	return numpy.sum(negLog)
LogLikelihoodSum.optimal = 'min'


def LogLikelihoodSumDrop5Percent(knownValues, predictedValues):
	"""
	Where predictedValues contains the likelihood of each point given the
	model we are testing, after removing the 5% least likely points, take
	the log of each likelihood and sum the results.

	"""
	dropped = filterLowest(predictedValues, .05)
	return LogLikelihoodSum(knownValues, dropped)
LogLikelihoodSumDrop5Percent.optimal = 'min'

def filterLowest(obj, toDrop=.05):
	obj = obj.copy()
	if obj.pointCount != 1 and obj.featureCount != 1:
		raise UML.exceptions.ArgumentException("Obj must be vector shaped")
	if obj.pointCount != 1:
		obj.transpose()

	obj.sortFeatures(0)

	if isinstance(toDrop, float):
		# we rely on this to convert via truncation to ensure we're including
		# as much as possible in the result
		start = int(obj.featureCount * toDrop)
	else:
		start = toDrop
	return obj.extractFeatures(start=start)


def testFilterLowest():
	raw = [8,16,4,32]
	test = UML.createData("Matrix", raw)

	# sorted result, correctly discards 2 elements
	assert filterLowest(test, 2).copyAs("pythonlist") == [[16,32]]
	# sorted result, correctly discards lowest quarter
	assert filterLowest(test, .25).copyAs("pythonlist") == [[8,16,32]]
	# sorted result, truncates .2 down to 0 elements
	assert filterLowest(test, .2).copyAs("pythonlist") == [[4,8,16,32]]

def test_LogPobSum():
	raw = [8,16,32]
	test = UML.createData("Matrix", raw)
	ret = LogLikelihoodSum(None, test)
	assert ret == -12

class KDEProbability(UML.customLearners.CustomLearner):
	learnerType = "unknown"

	def train(self, trainX, trainY, bandwidth=None):
		if not trainY is None:
			raise ValueError("This is an unsupervised method, trainY must be None")

		self.kde = scipy.stats.gaussian_kde(trainX, bw_method=bandwidth)

	def apply(self, testX):
		retType = testX.getTypeString()
		ret = UML.createData(retType, self.kde.evaluate(testX))
		ret.transpose()
		return ret


def cleanName(name):
	"""
	Given a string name, return a string with no leading/trailing whitespace
	or matched pairs of leading/trailing single/double quotes

	"""
	temp = name.strip()
	lastIndex = len(temp)-1
	leading = temp[0] == "'" or temp[0] == '"'
	ending = temp[lastIndex] == "'" or temp[lastIndex] == '"'

	if leading and ending:
		temp = temp[1:lastIndex]

	return temp		


def cleanFeatureNames(obj):
	"""
	Modifies object to have have feature names processed by the cleanName function.

	"""
	names = obj.getFeatureNames()
	clean = []
	for name in names:
		temp = cleanName(name)
		clean.append(temp)

	obj.setFeatureNames(clean)


def loadCategoryData(path):
	"""
	Return a double: a UML object with category and gender data point-indexed by
	question name; and a dict mapping category name to list of questions in that
	category	

	"""
	categories = UML.createData("List", path, featureNames=True, pointNames=False)
#	categories.show("before")

	cleanFeatureNames(categories)

	def cleanNamesInPoint(point):
		return [cleanName(point[0]), point[1], cleanName(point[2])]

	categories.transformEachPoint(cleanNamesInPoint)

#	categories.show("after")

	categoriesByQName = categories.copyAs("List")
	qs = categoriesByQName.extractFeatures("question")
	categoriesByQName.setPointNames(qs.copyAs('pythonlist', outputAs1D=True))
#	categoriesByQName.show('cat')

	namesByCategory = {}

	def collect(point):
		catName = point[0]
		qName = point[2]
		if catName not in namesByCategory:
			namesByCategory[catName] = []

		namesByCategory[catName].append(qName)

	categories.calculateForEachPoint(collect)
#	print namesByCategory

	return categoriesByQName, namesByCategory


def invertGenderScoring(obj):
	"""
	Given a feature vector shaped object containing 1 values to indicate
	a point is from a male and 0 values to indicate if a point is from
	a female, return an object where 0 is male and 1 is female.

	"""
	allOnes = numpy.ones((len(obj), 1))
	allOnes = UML.createData("Matrix", allOnes)
	ret = allOnes - obj
	return ret


def loadResponseData(path):

	responses = UML.createData("List", path, featureNames=True, pointNames=False)
	cleanFeatureNames(responses)

	nonBinary = responses.extractPoints([0,1,2,3,4,5])

	genderValueID = responses.getFeatureIndex("male1female0")
	genderValue = responses.extractFeatures(genderValueID)
	genderValue = genderValue.copyAs("Matrix")
	genderValue = invertGenderScoring(genderValue)
	genderValue.setFeatureName(0, "male0female1")

	firstCat = responses.getFeatureIndex('"agreeable" category')
	categoryScores = responses.extractFeatures(start=firstCat)
	categoryScores = categoryScores.copyAs("Matrix")

#	categoryScores.show('catScores')

	firstQID = responses.getFeatureIndex("I contradict others.")
	lastQID = responses.getFeatureIndex("I get stressed out.")
	responses = responses.extractFeatures(start=firstQID, end=lastQID)
	responses = responses.copyAs("Matrix")

	responses.appendFeatures(genderValue)

#	genderValue.show("gender", maxWidth=120, maxHeight=30, nameLength=15)
#	responses.show("Qs", maxWidth=120, maxHeight=30, nameLength=15)

	return responses, categoryScores


def randomize_splitTrainTest(obj, trainNum):
	obj.shufflePoints()
	train = obj.extractPoints(end=(trainNum-1))
#	print train.pointCount
	assert train.pointCount == trainNum

	test = obj
	return train, test



def checkFromFileCatScores(categoryScores, namesByCategory, categoriesByQName, responses):
	for category, (q0, q1, q2, q3) in namesByCategory.items():
		sub1 = generateSubScale(responses, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1])
		sub2 = generateSubScale(responses, q2, categoriesByQName[q2,1], q3, categoriesByQName[q3,1])
		fullCatScore = (sub1 + sub2) / 2.0
		roundTo1SigDig = partial(round, ndigits=1)
		roundedCatScoreRaw = map(roundTo1SigDig, fullCatScore)
		roundedCatScore = UML.createData("Matrix", roundedCatScoreRaw)
		roundedCatScore.transpose()

		copyName = '"' + category + '" category'
		fromFileScores = categoryScores.copyFeatures(copyName)

		diff = numpy.absolute((fromFileScores - roundedCatScore).copyAs("numpyarray", outputAs1D=True))
		assert numpy.all(diff <= 0.100000000000001)


def determineBestSubScores(namesByCategory, categoriesByQName, responses, genderValue):
	picked = {}

	for category, (q0, q1, q2, q3) in namesByCategory.items():
		q0Gender = categoriesByQName[q0,1]
		q1Gender = categoriesByQName[q1,1]
		q2Gender = categoriesByQName[q2,1]
		q3Gender = categoriesByQName[q3,1]

		q0q1 = generateSubScale(responses, q0, q0Gender, q1, q1Gender)
		q0q2 = generateSubScale(responses, q0, q0Gender, q2, q2Gender)
		q0q3 = generateSubScale(responses, q0, q0Gender, q3, q3Gender)
		q1q2 = generateSubScale(responses, q1, q1Gender, q2, q2Gender)
		q1q3 = generateSubScale(responses, q1, q1Gender, q3, q3Gender)
		q2q3 = generateSubScale(responses, q2, q2Gender, q3, q3Gender)

		q0q1Corr = scoreToGenderCorrelation(q0q1, genderValue)
		q0q2Corr = scoreToGenderCorrelation(q0q2, genderValue)
		q0q3Corr = scoreToGenderCorrelation(q0q3, genderValue)
		q1q2Corr = scoreToGenderCorrelation(q1q2, genderValue)
		q1q3Corr = scoreToGenderCorrelation(q1q3, genderValue)
		q2q3Corr = scoreToGenderCorrelation(q2q3, genderValue)

		allCorr = [q0q1Corr, q0q2Corr, q0q3Corr, q1q2Corr, q1q3Corr, q2q3Corr]
#		if category == "non resilient to illness":
#			print allCorr

		best = max(allCorr)
		if best == q0q1Corr:
			pickedQs = (q0, q1)
		elif best == q0q2Corr:
			pickedQs = (q0, q2)
		elif best == q0q3Corr:
			pickedQs = (q0, q3)
		elif best == q1q2Corr:
			pickedQs = (q1, q2)
		elif best == q1q3Corr:
			pickedQs = (q1, q3)
		else:
			pickedQs = (q2, q3)

		picked[category] = pickedQs

	return picked


def verifyAvg(picked, responses, genderValue):
	def extractFemale(point):
		pID = responses.getPointIndex(point.getPointName(0))
		return genderValue[pID] == 1

	toSplit = responses.copy()
	femalePoints = toSplit.extractPoints(extractFemale)
	malePoints = toSplit

	for cat, (q1, q2) in picked.items():
		q1Gender = categoriesByQName[q1,1]
		q2Gender = categoriesByQName[q2,1]

		fSubscale = generateSubScale(femalePoints, q1, q1Gender, q2, q2Gender)
		fAvg = fSubscale.featureStatistics("mean")[0]
		mSubscale = generateSubScale(malePoints, q1, q1Gender, q2, q2Gender)
		mAvg = mSubscale.featureStatistics("mean")[0]

#		print str(fAvg) + " " + str(mAvg)
		assert fAvg > mAvg

def verifyBandwidthSelectionWorks(responses, genderValue):
	def extractFemale(point):
		pID = responses.getPointIndex(point.getPointName(0))
		return genderValue[pID] == 1

	toSplit = responses.copy()
	femalePoints = toSplit.extractPoints(extractFemale)
	malePoints = toSplit
	numMale = malePoints.pointCount
	numFemale = femalePoints.pointCount

	# TRIAL: normal distributions
	muM, sigmaM = -5, 3
	muF, sigmaF = 5, 3
	genDataM = UML.createData("Matrix", numpyRandom.normal(muM, sigmaM, numMale).reshape(numMale,1))
	genDataF = UML.createData("Matrix", numpyRandom.normal(muF, sigmaF, numFemale).reshape(numFemale,1))

#	bw = tuple([.02 + i*.02 for i in xrange(25)])
	bwBaseM = silver_factor(genDataM.pointCount, genDataM.featureCount)
	bw = [bwBaseM * (1.1 ** i) for i in xrange(-15,15)]
	mfolds = 10
	mAll = UML.crossValidateReturnAll("custom.KDEProbability", genDataM, None, bandwidth=bw, numFolds=mfolds, performanceFunction=LogLikelihoodSum)
	mBW = cvUnpackBest(mAll, False)[0]['bandwidth']

	bwBaseF = silver_factor(genDataM.pointCount, genDataM.featureCount)
	bw = [bwBaseF * (1.1 ** i) for i in xrange(-15,15)]
	ffolds = 10
	fAll = UML.crossValidateReturnAll("custom.KDEProbability", genDataF, None, bandwidth=bw, numFolds=ffolds, performanceFunction=LogLikelihoodSum)
	fBW = cvUnpackBest(fAll, False)[0]['bandwidth']

	opts = {}
	opts['fileName'] = None
	opts['title'] = str(mBW) + " Generated bandwidth trial " + str(fBW)
	opts['xlabel'] = ""
	opts['showPoints'] = True
	opts['xLimits'] = (-10, 10)
	opts['yLimits'] = (0, .15)
	plotDualWithBlendFill(genDataM, genDataF, None, None, **opts)

	baseX = [(0.5 * x) - 10 for x in xrange(0,41)]
	plt.plot(baseX, scipy.stats.norm.pdf(baseX, muM, sigmaM), linewidth=2, color='blue')
	plt.plot(baseX, scipy.stats.norm.pdf(baseX, muF, sigmaF), linewidth=2, color='red')

#	plt.show()
	filename = "/home/tpburns/gimbel_tech/data/gender/2nd_round_trial/known_trial_normal.png"
	plt.savefig(filename)
	plt.close()


	# TRIAL: combination normal distributions
	muM1, sigmaM1 = -7, 2
	muM2, sigmaM2 = -2, 4
	m1 = numpyRandom.normal(muM1, sigmaM1, numMale)
	m2 = numpyRandom.normal(muM2, sigmaM2, numMale)
	mSelected = pythonRandom.sample(numpy.append(m1,m2), numMale)
	genDataM = UML.createData("Matrix", mSelected)
	genDataM.transpose()
	
	muF1, sigmaF1 = 0, 3
	muF2, sigmaF2 = 8, 3
	f1 = numpyRandom.normal(muF1, sigmaF1, numFemale)
	f2 = numpyRandom.normal(muF2, sigmaF2, numFemale)
	fSelected = pythonRandom.sample(numpy.append(f1,f2), numFemale)
	genDataF = UML.createData("Matrix", fSelected)
	genDataF.transpose()

	bwBaseM = silver_factor(genDataM.pointCount, genDataM.featureCount)
	bw = [bwBaseM * (1.1 ** i) for i in xrange(-15,15)]
	mfolds = 10
	mAll = UML.crossValidateReturnAll("custom.KDEProbability", genDataM, None, bandwidth=bw, numFolds=mfolds, performanceFunction=LogLikelihoodSum)
	mBW = cvUnpackBest(mAll, False)[0]['bandwidth']

	bwBaseF = silver_factor(genDataM.pointCount, genDataM.featureCount)
	bw = [bwBaseF * (1.1 ** i) for i in xrange(-15,15)]
	ffolds = 10
	fAll = UML.crossValidateReturnAll("custom.KDEProbability", genDataF, None, bandwidth=bw, numFolds=ffolds, performanceFunction=LogLikelihoodSum)
	fBW = cvUnpackBest(fAll, False)[0]['bandwidth']

	opts = {}
	opts['fileName'] = None
	opts['title'] = str(mBW) + " Generated bandwidth trial " + str(fBW)
	opts['xlabel'] = ""
	opts['showPoints'] = True
	opts['xLimits'] = (-10, 10)
	opts['yLimits'] = (0, .15)
	plotDualWithBlendFill(genDataM, genDataF, mBW, fBW, **opts)

	baseX = [(0.5 * x) - 10 for x in xrange(0,41)]

	def pdfM(vals):
		return (scipy.stats.norm.pdf(vals, muM1, sigmaM1) / 2.0) + (scipy.stats.norm.pdf(vals, muM2, sigmaM2) / 2.0)
	
	def pdfF(vals):
		return (scipy.stats.norm.pdf(vals, muF1, sigmaF1) / 2.0) + (scipy.stats.norm.pdf(vals, muF2, sigmaF2) / 2.0)

	plt.plot(baseX, pdfM(baseX), linewidth=2, color='blue')
	plt.plot(baseX, pdfF(baseX), linewidth=2, color='red')

#	plt.show()
	filename = "/home/tpburns/gimbel_tech/data/gender/2nd_round_trial/known_trial_combined.png"
	plt.savefig(filename)
	plt.close()




def outputFile_SelectedQsPerCategory(outPath, categoriesByQName, picked):
	raw = []
	for point in categoriesByQName.pointIterator():
		question = point.getPointName(0)
		category = point[0]

		gender = 1 if point[1] == 'female' else -1
		if question not in picked[category]:
			gender = 0
		
		newPoint = [question, gender, category]
		raw.append(newPoint)

	fnames = ["Questions", "Selected", "Category"]
	toOutput = UML.createData("List", raw, featureNames=fnames)
#	print sum(ret.featureView(1))
#	toOutput.show("", maxWidth=120)

	toOutput.writeFile(outPath)


def outputFile_subCategoryCorrelationWithGender(outPath, picked, categoriesByQName, responses,
		genderValue):
	raw = []
	names = []
	for category, (q1, q2) in picked.items():
		q1Gender = categoriesByQName[q1,1]
		q2Gender = categoriesByQName[q2,1]
		subscale = generateSubScale(responses, q1, q1Gender, q2, q2Gender)
		corr = scoreToGenderCorrelation(subscale, genderValue)
		raw.append([corr])
		names.append(category)

	toOut = UML.createData("List", raw, pointNames=names)
	toOut.writeFile(outPath)

def outputFile_CategoryCorrelationWithGender(outPath, namesByCategory,
		categoriesByQName, responses):

	genderValue = responses.copyFeatures("male0female1")

	raw = []
	names = []
	for category, (q0, q1, q2, q3) in namesByCategory.items():
		sub1 = generateSubScale(responses, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1])
		sub2 = generateSubScale(responses, q2, categoriesByQName[q2,1], q3, categoriesByQName[q3,1])
		fullCatScore = (sub1 + sub2) / 2.0

		corr = scoreToGenderCorrelation(fullCatScore, genderValue)
		pval = scoreToGenderPValues(fullCatScore, genderValue)
		raw.append([pval[0], pval[1]])
		names.append(category)

	toOut = UML.createData("List", raw, pointNames=names)
	toOut.writeFile(outPath)


def printCategoryCorrelationToGender(namesByCategory, categoriesByQName, responses):
	for category, (q0, q1, q2, q3) in namesByCategory.items():
		sub1 = generateSubScale(responses, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1])
		sub2 = generateSubScale(responses, q2, categoriesByQName[q2,1], q3, categoriesByQName[q3,1])
		fullCatScore = (sub1 + sub2) / 2.0
		roundTo1SigDig = partial(round, ndigits=1)
		roundedCatScoreRaw = map(roundTo1SigDig, fullCatScore)
		roundedCatScore = UML.createData("Matrix", roundedCatScoreRaw)
		roundedCatScore.transpose()

		fullGender = responses.copyFeatures("male0female1")
		print category + ": " + str(scoreToGenderCorrelation(fullCatScore, fullGender))
		print category + ": " + str(scoreToGenderCorrelation(roundedCatScore, fullGender))
		print ""


def scoreToGenderCorrelation(scores, genders):
	scores.appendFeatures(genders)
	corr = scores.featureSimilarities("correlation")
	scores.extractFeatures(1)
	return corr[0,1]

def scoreToGenderPValues(scores, genders):
	scores = scores.copyAs("numpyarray", outputAs1D=True)
	genders = genders.copyAs("numpyarray", outputAs1D=True)
	pvals = scipy.stats.pearsonr(scores, genders)
	return pvals


def generateSubScale(data, qA_ID, qA_Gender, qB_ID, qB_Gender):
	qA = data.copyFeatures(qA_ID)
	qB = data.copyFeatures(qB_ID)
	qA.setFeatureNames(None)
	qB.setFeatureNames(None)

	qA = -qA if qA_Gender == 'male' else qA
	qB = -qB if qB_Gender == 'male' else qB

	return (qA + qB) / 2.0


def cvUnpackBest(resultsAll, maximumIsBest):
	bestArgumentAndScoreTuple = None
	for curResultTuple in resultsAll:
		curArgument, curScore = curResultTuple
		#if curArgument is the first or best we've seen: 
		#store its details in bestArgumentAndScoreTuple
		if bestArgumentAndScoreTuple is None:
			bestArgumentAndScoreTuple = curResultTuple
		else:
			if (maximumIsBest and curScore > bestArgumentAndScoreTuple[1]):
				bestArgumentAndScoreTuple = curResultTuple
			if ((not maximumIsBest) and curScore < bestArgumentAndScoreTuple[1]):
				bestArgumentAndScoreTuple = curResultTuple

	return bestArgumentAndScoreTuple



def addNoiseToResponses(responses):
#	print responses[0,0]
#	print responses[1,1]

	size = (responses.pointCount, responses.featureCount)

	npr = UML.randomness.numpyRandom

	# gives us random values in the range [0, 0.1)
	positiveNoise = (npr.rand(size[0], size[1])) / 10
	# randomly generates a -1 or a 1.
	signAdjustment = (npr.randint(2, size=size) * 2) - 1

	noise = positiveNoise * signAdjustment
	noiseObj = UML.createData("Matrix", noise)

	noiseObj += responses
	
#	print noiseObj[0,0]
#	print noiseObj[1,1]
	return noiseObj



def bandwidthTrials(picked, categoriesByQName, responses, genderValue, LOOfolding=False):
	def extractFemale(point):
		pID = responses.getPointIndex(point.getPointName(0))
		return genderValue[pID] == 1

	toSplit = responses.copy()
	femalePoints = toSplit.extractPoints(extractFemale)
	malePoints = toSplit

	num = 0
	mResults = {}
	fResults = {}
	for cat, (q1, q2) in picked.items():
		q1Gender = categoriesByQName[q1,1]
		q2Gender = categoriesByQName[q2,1]

		mSubscale = generateSubScale(malePoints, q1, q1Gender, q2, q2Gender)
		fSubscale = generateSubScale(femalePoints, q1, q1Gender, q2, q2Gender)
#		print mSubscale.pointCount
#		print mSubscale.featureCount
#		mSubscale = generateSubScale(malePoints, q1, q1Gender, q2, q2Gender).extractPoints(end=10)
#		fSubscale = generateSubScale(femalePoints, q1, q1Gender, q2, q2Gender).extractPoints(end=10)

#		bw = tuple([.02 + i*.02 for i in xrange(25)])
		bw = tuple([.5 - i*.02 for i in xrange(25)])
#		bw = (.1,.2,.3)

		print cat
		if LOOfolding:
			mfolds = mSubscale.pointCount
		else:
			mfolds = 2
#		print mfolds

#		perfFunc = LogLikelihoodSum
#		perfFunc = LogLikelihoodSumDrop5Percent
		perfFunc = makePowerOfAdjustedLogLikelihoodSum(1)

		boundary = "********************************************************************************"
		UML.logger.active.humanReadableLog.logMessage(boundary + "\n" + cat)

		mAll = UML.crossValidateReturnAll("custom.KDEProbability", mSubscale, None, bandwidth=bw, numFolds=mfolds, performanceFunction=perfFunc)
		mBest = cvUnpackBest(mAll, False)
		mResults[cat] = mBest[0]['bandwidth']
#		print "MSCALE"
#		print mBest
#		print mAll

		if LOOfolding:
			ffolds = fSubscale.pointCount
		else:
			ffolds = 2
#		print ffolds
		fAll = UML.crossValidateReturnAll("custom.KDEProbability", fSubscale, None, bandwidth=bw, numFolds=ffolds, performanceFunction=perfFunc)
		fBest = cvUnpackBest(fAll, False)
		fResults[cat] = fBest[0]['bandwidth']
#		print "FSCALE"
#		print fBest
#		print fAll

	print ""
	print mResults
	print fResults
	return mResults, fResults



def fixedBandwidth_noisy_drop5percent():
	mbw = {	}

	fbw = {	}

	return mbw,fbw

def fixedBandwidth_drop5percent():
	mbw = {'agreeable':					0.28,	'emotionally aware': 		0.22, 
			'non eloquent': 			0.04,	'non image conscious':		0.24, 
			'non manipulative': 		0.14,	'altruistic':				0.30,
			'power avoidant':			0.16,	'non resilient to stress':	0.12,
			'non resilient to illness': 0.16,	'annoyable':				0.22, 
			'complexity avoidant':		0.1,	'warm':						0.08, 
			'optimistic':				0.14,	'non sexual':				0.04,
			'risk avoidant':			0.14,	'thin skinned':				0.16,
			'forgiving':				0.16,	'worried':					0.16,
			'talkative':				0.14,	'ordinary':					0.06,
			'empathetic':				0.06}

	fbw = {'agreeable':					0.16,	'emotionally aware':		0.1,
			'non eloquent':				0.18,	'non image conscious':		0.16,
			'non manipulative':			0.08,	'altruistic':				0.04,
			'power avoidant':			0.14,	'non resilient to stress':	0.06,
			'non resilient to illness': 0.18,	'annoyable':				0.26,
			'complexity avoidant':		0.28,	'warm':						0.04,
			'optimistic':				0.08,	'non sexual':				0.04,
			'risk avoidant':			0.12,	'thin skinned':				0.1,
			'forgiving':				0.04,	'worried':					0.04,
			'talkative':				0.1,	'ordinary':					0.1,
			'empathetic':				0.04}

	return mbw,fbw


def fixedBandwidth_noisy_all():
	mbw = {	'agreeable': 				0.4, 	'emotionally aware': 		0.26,
			'non eloquent': 			0.1, 	'non image conscious': 		0.24,
			'non manipulative': 		0.20,	'altruistic': 				0.36,
			'power avoidant': 			0.34,	'non resilient to stress': 	0.12,
			'non resilient to illness': 0.22, 	'annoyable': 				0.30,
			'complexity avoidant': 		0.14,	'warm': 					0.30,
			'optimistic': 				0.32, 	'non sexual': 				0.12,
			'risk avoidant': 			0.26, 	'thin skinned': 			0.16,
			'forgiving': 				0.18, 	'worried':					0.18,
			'talkative': 				0.14, 	'ordinary': 				0.4,
			'empathetic': 				0.1}

	fbw = {	'agreeable': 				0.3, 	'emotionally aware': 		0.20,
			'non eloquent': 			0.22, 	'non image conscious': 		0.28,
			'non manipulative': 		0.16, 	'altruistic': 				0.24,
			'power avoidant': 			0.16,	'non resilient to stress': 	0.24,
			'non resilient to illness': 0.22, 	'annoyable': 				0.28,
			'complexity avoidant': 		0.36, 	'warm': 					0.16,
			'optimistic': 				0.30, 	'non sexual': 				0.08,
			'risk avoidant': 			0.12, 	'thin skinned': 			0.16,
			'forgiving': 				0.12, 	'worried': 					0.24,
			'talkative': 				0.14, 	'ordinary': 				0.2,
			'empathetic': 				0.30}

	return mbw,fbw

def fixedBandwidth_all():
	mbw = { 'agreeable': 				0.4,	'emotionally aware':		0.24,
			'non eloquent': 			0.08, 	'non image conscious': 		0.24,
			'non manipulative': 		0.20, 	'altruistic': 				0.36,
			'power avoidant': 			0.34, 	'non resilient to stress': 	0.1,
			'non resilient to illness': 0.22, 	'annoyable': 				0.32,
			'complexity avoidant': 		0.16, 	'warm': 					0.28,
			'optimistic': 				0.32, 	'non sexual': 				0.12,
			'risk avoidant': 			0.28, 	'thin skinned': 			0.18,
			'forgiving': 				0.18, 	'worried': 					0.18,
			'talkative': 				0.14,	'ordinary': 				0.4,
			'empathetic': 				0.1}
	
	fbw = { 'agreeable': 				0.30, 	'emotionally aware': 		0.20,
			'non eloquent': 			0.24, 	'non image conscious':		0.28,
			'non manipulative':			0.16, 	'altruistic': 				0.24,
			'power avoidant': 			0.18, 	'non resilient to stress':	0.24,
			'non resilient to illness': 0.22, 	'annoyable':				0.28,
			'complexity avoidant':		0.36, 	'warm': 					0.18,
			'optimistic': 				0.28, 	'non sexual': 				0.1,
			'risk avoidant': 			0.12, 	'thin skinned': 			0.16,
			'forgiving': 				0.12, 	'worried':					0.24,
			'talkative': 				0.14, 	'ordinary': 				0.18,
			'empathetic': 				0.28}

	return mbw,fbw

def fixedBandwidth_handAdjusted():
	mbw = { 'agreeable': 				0.23,	'emotionally aware':		0.21,
			'non eloquent': 			0.17, 	'non image conscious': 		0.21,
			'non manipulative': 		0.20, 	'altruistic': 				0.23,
			'power avoidant': 			0.22, 	'non resilient to stress': 	0.18,
			'non resilient to illness': 0.20, 	'annoyable': 				0.22,
			'complexity avoidant': 		0.18, 	'warm': 					0.20,
			'optimistic': 				0.21, 	'non sexual': 				0.17,
			'risk avoidant': 			0.21, 	'thin skinned': 			0.19,
			'forgiving': 				0.19, 	'worried': 					0.19,
			'talkative': 				0.18,	'ordinary': 				0.22,
			'empathetic': 				0.17}
	
	fbw = { 'agreeable': 				0.2, 	'emotionally aware': 		0.18,
			'non eloquent': 			0.19, 	'non image conscious':		0.19,
			'non manipulative':			0.17, 	'altruistic': 				0.18,
			'power avoidant': 			0.18, 	'non resilient to stress':	0.18,
			'non resilient to illness': 0.18, 	'annoyable':				0.2,
			'complexity avoidant':		0.21, 	'warm': 					0.17,
			'optimistic': 				0.19, 	'non sexual': 				0.15,
			'risk avoidant': 			0.17, 	'thin skinned': 			0.17,
			'forgiving': 				0.16, 	'worried':					0.18,
			'talkative': 				0.17, 	'ordinary': 				0.18,
			'empathetic': 				0.18}

	return mbw,fbw


def fixedBandwidth_same(namesByCategory, mVal, fVal):
	mbw = copy.copy(namesByCategory)
	for k in mbw.keys():
		mbw[k] = mVal

	fbw = copy.copy(namesByCategory)
	for k in fbw.keys():
		fbw[k] = fVal

	return mbw, fbw


def collateBW():
	mbw1,fbw1 = fixedBandwidth_drop5percent()
	mbw2,fbw2 = fixedBandwidth_noisy_all()
	mbw3,fbw3 = fixedBandwidth_all()

	print "Male"
	mAvg = 0
	for k in mbw1.keys():
		v1 = mbw1[k]
		v2 = mbw2[k]
		v3 = mbw3[k]
		avgV = (v1 + v2 + v3) / 3
		avgS = '%.3f' % round(avgV, 3)
		mAvg += avgV
		print (k + ": ").ljust(30) + str(v1) + " " + str(v2) + " " + str(v3) + " " + avgS

	print mAvg / len(mbw1)

	print "\n Female"
	fAvg = 0
	for k in fbw1.keys():
		v1 = fbw1[k]
		v2 = fbw2[k]
		v3 = fbw3[k]
		avgV = (v1 + v2 + v3) / 3
		avgS = '%.3f' % round(avgV, 3)
		fAvg += avgV
		print (k + ": ").ljust(30) + str(v1) + " " + str(v2) + " " + str(v3) + " " + avgS

	print fAvg / len(fbw1)

#def generatePlotsRange(picked, categoriesByQName, responses, genderValue, outDir, bw):
#	for i in xrange(-2, 3):
#		for k,v in bw[0].items():
#			pass


def generatePlots(picked, categoriesByQName, responses, genderValue, outDir, bw):
	def extractFemale(point):
		pID = responses.getPointIndex(point.getPointName(0))
		return genderValue[pID] == 1

	toSplit = responses.copy()
	femalePoints = toSplit.extractPoints(extractFemale)
	malePoints = toSplit

	num = 0
	for cat, (q1, q2) in picked.items():
		q1Gender = categoriesByQName[q1,1]
		q2Gender = categoriesByQName[q2,1]

		fSubscale = generateSubScale(femalePoints, q1, q1Gender, q2, q2Gender)  # .extractPoints(end=20)
		mSubscale = generateSubScale(malePoints, q1, q1Gender, q2, q2Gender)  # .extractPoints(end=20)

#		fSubscale.show("F", maxWidth=None, maxHeight=None)

#		kdeAndHistogramPlot(mSubscale, title=cat+" Ma", show=True)
#		kdeAndHistogramPlot(fSubscale, title=cat+" Fem", show=True)
		fileName = os.path.join(outDir, cat)
		opts = {}
		opts['fileName'] = fileName
		opts['show'] = False
		opts['title'] = str(bw[0][cat]) + " | " + cat + " | " + str(bw[1][cat])
#		opts['title'] = cat
		opts['xlabel'] = ""
		opts['showPoints'] = False
		opts['xLimits'] = (-10, 10)
		opts['yLimits'] = (0, .18)
		plotDualWithBlendFill(mSubscale, fSubscale, bw[0][cat], bw[1][cat], **opts)
#		plotDualWithBlendFill(mSubscale, fSubscale, None, None, **opts)
#		if num > 0:
#		assert False
		num += 1


if __name__ == '__main__':
	#sys.exit(0)
	TRAIN_NUMBER = 300
	PRINT_FULL_CAT_CORR = False
	OUTPUT_FULL_CAT_CORR_AND_PVALS = False
	VERIFY_FROMFILE_CATEGORYSCORES = False
	VERIFY_MEANS_ORDERING_OF_SUBSCORES = True
	VERIFY_BANDWIDTH_SELECTION_FEASIBLE = False
	OUTPUT_SEL_QS_PER_CAT = False
	OUTPUT_SUBSCORE_CAT_CORR = False

	ADD_NOISE_TO_DATA = False
	RUN_BANDWIDTH_TRIALS = True
	OUTPUT_PLOTS = False

	UML.registerCustomLearner('Custom', KDEProbability)

	import time
	print time.asctime(time.localtime())

	sourceDir = sys.argv[1]
	path_categories = os.path.join(sourceDir, "question_categories.csv")
	path_responses = os.path.join(sourceDir, "question_data.csv")
	outPath_selected = os.path.join(sourceDir, "question_selected.csv")
	outPath_subscore_corr = os.path.join(sourceDir, "subscore_category_correlation.csv")
	outPath_fullcat_corr_pval = os.path.join(sourceDir, "category_to_gender_correlation_and_pvals.csv")
	outDir_plots = os.path.join(sourceDir, "plots")

	# Load data from the provided paths
	categoriesByQName, namesByCategory = loadCategoryData(path_categories)
	responses, categoryScores = loadResponseData(path_responses)

	if PRINT_FULL_CAT_CORR:
		printCategoryCorrelationToGender(namesByCategory, categoriesByQName, responses)
	if OUTPUT_FULL_CAT_CORR_AND_PVALS:
		outputFile_CategoryCorrelationWithGender(outPath_fullcat_corr_pval, namesByCategory, categoriesByQName, responses)
	if VERIFY_FROMFILE_CATEGORYSCORES:
		checkFromFileCatScores(categoryScores, namesByCategory, categoriesByQName, responses)

	# remove the categories we have determined do not significantly distinguish between
	# genders according to p-value
#	if OMIT_NON_SIGNIFICANT_CATEGORIES:
#		pass

	# Split gender / response data for subscore selection training and visualziation
	testFraction = float(responses.pointCount - TRAIN_NUMBER) / responses.pointCount
	responseTrain, genderTrain, responseTest, genderTest = responses.trainAndTestSets(testFraction, "male0female1")
	selected = determineBestSubScores(namesByCategory, categoriesByQName, responseTrain, genderTrain)

	if VERIFY_MEANS_ORDERING_OF_SUBSCORES:
		verifyAvg(selected, responseTest, genderTest)
	if VERIFY_BANDWIDTH_SELECTION_FEASIBLE:
		verifyBandwidthSelectionWorks(responseTest, genderTest)
	if OUTPUT_SEL_QS_PER_CAT:
		outputFile_SelectedQsPerCategory(outPath_selected, categoriesByQName, selected)
	if OUTPUT_SUBSCORE_CAT_CORR:
		outputFile_subCategoryCorrelationWithGender(outPath_subscore_corr, selected, categoriesByQName, responseTest, genderTest)

	# responses were only to one digit, which makes them clump for bandwidth trials.
	# add some noise to loosen things up without going outside the range of what would
	# round to the given value
	if ADD_NOISE_TO_DATA:
		responseTest_noisy = addNoiseToResponses(responseTest)
	else:
		responseTest_noisy = responseTest
	
	if RUN_BANDWIDTH_TRIALS:
		mBw, fBw = bandwidthTrials(selected, categoriesByQName, responseTest_noisy, genderTest, False)
	else:
		mBw, fBw = fixedBandwidth_same(namesByCategory, .5, .5)

	if OUTPUT_PLOTS:
		generatePlots(selected, categoriesByQName, responseTest_noisy, genderTest, outDir_plots, (mBw, fBw))

	print time.asctime(time.localtime())
	
	pass  # EOF marker
