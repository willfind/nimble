import pdb
import sys
import numpy
import scipy
import os.path
import functools
from functools import partial

KDE_HELPER_PATH = "/home/tpburns/Dropbox/ML_intern_tpb/python_workspace/kdePlotting"
ORIG_HELPER_PATH = "/home/tpburns/Dropbox/ML_intern_tpb/python_workspace/"
sys.path.append(KDE_HELPER_PATH)
sys.path.append(ORIG_HELPER_PATH)
from fancyHistogram import kdeAndHistogramPlot
from multiplot import plotDualWithBlendFill

from allowImports import boilerplate
boilerplate()

import UML
from UML.examples.gender.gender_visualization_bandwidth import *



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

def formatCategoryName(name):
	"""
	Given an unpadded string, capitalize all of it's words, and connect the
	prefix 'non' to its following word with a dash. To be used on category names,
	not questions.
	"""
	name = ' '.join(map(str.capitalize, name.split(' ')))

	if name.startswith('Non '):
		name = 'Non-' + name[4:]

	return name

def cleanFeatureNames(obj):
	"""
	Modifies given object to have have feature names processed by the cleanName function.

	"""
	names = obj.getFeatureNames()
	clean = []
	for name in names:
		temp = cleanName(name)
		clean.append(temp)

	obj.setFeatureNames(clean)


def loadCategoryData(path):
	"""
	Load the metadata for categories, questions, and gender scale from the given path.
	Return a double: a UML object with category and gender data point-indexed by
	question name; and a dict mapping category name to list of questions in that
	category. 

	"""
	categories = UML.createData("List", path, featureNames=True, pointNames=False)
#	categories.show("before")

	cleanFeatureNames(categories)

	def cleanAndFormatNamesInPoint(point):
		return [formatCategoryName(cleanName(point[0])), point[1], cleanName(point[2])]

	categories.transformEachPoint(cleanAndFormatNamesInPoint)

#	categories.show("after", maxWidth=None, maxHeight=None)

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

	# drop non-binary genders
	responses.extractPoints([0,1,2,3,4,5])

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


def checkFromFileCatScores(categoryScores, namesByCategory, categoriesByQName, responses):
	"""
	The response data set also included precalculated category subscores. This confirms
	those values by checking our independantly derived scores.

	"""
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


def removeProblemCategories(categoriesByQName, namesByCategory, responses):
	"""
	Remove the categories we have determined do not significantly distinguish between
	genders according to p-value or are too closely related to another, more expressive
	category. Specifically we remove: "annoyable", "non resilient to illness",
	"non resiliant to stress", "non image conscious", "optimistic", and "talkative"

	"""
	# remove from categoriesByQName, a UML object where the category is the 0th value
	# of each point
	beforePointCount = categoriesByQName.pointCount
	catToRemove = ["Annoyable", "Non-Resilient To Illness", "Non-Resilient To Stress", "Non-Image Conscious", "Optimistic", "Talkative"]

	def removeFunc(point):
		return point[0] in catToRemove

	categoriesByQName.extractPoints(removeFunc)
	afterPointCount = categoriesByQName.pointCount
	assert beforePointCount - afterPointCount == 24

	# Remove from responses, where each feature is scores for a particular question
	# (we need the contents of namesByCategory in order to find which quesiton is
	# in which category, so this removal happens first)
	qsToRemove = []
	for cat,qs in namesByCategory.items():
		if cat in catToRemove:
			qsToRemove += qs

	responsesPCBefore = responses.featureCount
	responses.extractFeatures(qsToRemove)
	responsesPCAfter = responses.featureCount
	assert responsesPCBefore - responsesPCAfter == 24

	# Remove from namesByCategory, a dict mapping category names to lists of questions
	nbcBefore = len(namesByCategory)
	for toRem in catToRemove:
		del namesByCategory[toRem]
	nbcAfter = len(namesByCategory)
	assert nbcBefore - nbcAfter == 6


def removeProblemQuestions(categoriesByQName, namesByCategory, responses):
	"""
	Remove those questions we have since deem to be poorly worded, or overlapping
	too strongly with questions in other categories. Specifically:
	"I get overwhelmed by difficult challenges.", "I freeze up under pressure.",
	"I need to plan out what I'm going to say in high pressure situations.",
	"I believe that I am better than others.", and
	"I would enjoy having multiple sexual partners if I were single."

	"""
	qsToRemove = ["I get overwhelmed by difficult challenges.",
					"I freeze up under pressure.",
					"I need to plan out what I'm going to say in high pressure situations.",
					"I believe that I am better than others.",
					"I would enjoy having multiple sexual partners if I were single."]

	# remove from categoriesByQName, a UML object where questions are point names
	beforePointCount = categoriesByQName.pointCount
	categoriesByQName.extractPoints(qsToRemove)
	afterPointCount = categoriesByQName.pointCount
	assert beforePointCount - afterPointCount == 5


	# Remove from namesByCategory, a dict mapping category names to lists of questions
	removed = 0
	for cat, qs in namesByCategory.items():
		keep = []
		for q in qs:
			if q not in qsToRemove:
				keep.append(q)
		removed += len(qs) - len(keep)
		namesByCategory[cat] = keep

	assert removed == 5


	# Remove from responses, where each feature is scores for a particular question
	responsesPCBefore = responses.featureCount
	responses.extractFeatures(qsToRemove)
	responsesPCAfter = responses.featureCount
	assert responsesPCBefore - responsesPCAfter == 5


def determineBestSubScores(namesByCategory, categoriesByQName, responses, genderValue):
	"""
	Out of the four questions for each category, pick those two that have the subscore
	most correlated with gender. These two questions are recorded in a dictionary,
	mapping category name to a tuple of question names.

	"""
	picked = {}

	for category, qs in namesByCategory.items():
		gender = [categoriesByQName[q,1] for q in qs]

		pairwiseScale = []
		mapping = []
		for i,q in enumerate(qs):
			for j,u in enumerate(qs[i+1:]):
				curr = generateSubScale(responses, q, gender[i], u, gender[j+i+1])
				pairwiseScale.append(curr)
				mapping.append((q,u))

		scoreCorrGenPartial = functools.partial(scoreToGenderCorrelation, genders=genderValue)
		allCorr = map(scoreCorrGenPartial, pairwiseScale)
		best = max(allCorr)

		for i,val in enumerate(allCorr):
			if val == best:
				pickedQs = mapping[i]

		picked[category] = pickedQs

	return picked


def verifyGenderAvgerageOrdering(picked, responses, genderValue):
	"""
	Verify that when we generate subscores, the average value for females is
	always higher than the average value for males. This is to ensure consistency
	when generating visualziations of the data.

	"""
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

def outputSelectedCategoryCorrelationToGender(responses, gender, selected, categoriesByQName, out=None):
	collected = []
	cats = []
	for category, (q0, q1) in selected.items():
		cats.append(category)
		score = generateSubScale(responses, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1])

		collected.append(scoreToGenderCorrelation(score, gender))

	collected = UML.createData("Matrix", collected, featureNames=cats)

	if out is None:
		collected.show("Selected Category Score Correlation to Gender", maxHeight=None, maxWidth=None, nameLength=50)
	else:
		collected.writeFile(out)




def printSelectedCategoryCorrelationMatrix(responseTest, selected, categoriesByQName, outFile=None):
	collected = None
	for category, (q0, q1) in selected.items():
		sub = generateSubScale(responses, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1])
		sub.setFeatureName(0, category)
		if collected is None:
			collected = sub
		else:
			collected.appendFeatures(sub)
	
	corrs = collected.featureSimilarities('correlation')
#	corrs.show("Selected Category Correlation Matrix", maxHeight=None, maxWidth=None, nameLength=25)
	if outFile is not None:
		corrs.writeFile(outFile)

def printSelectedQuestionCorrelationMatrix(responses, selected, outFile=None):
	collected = None
	for category, qs in selected.items():
		for q in qs:
			sub = responses.copyFeatures(q)
			sub.setFeatureName(0, q)
			if collected is None:
				collected = sub
			else:
				collected.appendFeatures(sub)
	
	corrs = collected.featureSimilarities('correlation')
#	corrs.show("Selected Question Correlation Matrix", maxHeight=None, maxWidth=None, nameLength=25)
	if outFile is not None:
		corrs.writeFile(outFile)


def printSelectedQuestionCorrelationInCategory(responses, selected, outFile=None):
	collected = None
	for category, (q1,q2) in selected.items():
		sub1 = responses.copyFeatures(q1)
		sub2 = responses.copyFeatures(q2)
		sub1.appendFeatures(sub2)

		corr = sub1.featureSimilarities('correlation')
		corr = corr.extractPoints(0).extractFeatures(1)
		corr.setFeatureName(0,category)
		corr.setPointName(0, 'InCat_QtoQ_corr')
		if collected is None:
			collected = corr
		else:
			collected.appendFeatures(corr)
	
	collected.show("Selected Question Correlation Matrix", maxHeight=None, maxWidth=None, nameLength=25)
	if outFile is not None:
		collected.writeFile(outFile)



def printSelectedQuestionToSelectedCategoryCorrelation(responses, selected, categoriesByQName, outPath):
	collected = None
	for category, qs in selected.items():
		sub = generateSubScale(responses, qs[0], categoriesByQName[qs[0],1], qs[1], categoriesByQName[qs[1],1])
		for qName in qs:
			q = responses.copyFeatures(qName)

			q.appendFeatures(sub)
			corr = q.featureSimilarities("correlation")
			corr.setPointName(0, qName)
			corr.setFeatureName(1, 'Q to Cat Corr')
			corr = corr.view(0,0,1,1)
			if collected is None:
				collected = UML.createData("Matrix", [], featureNames=['Q to Cat Corr'])
			collected.appendPoints(corr)

	collected = abs(collected)
#	collected.show("Selected Question To Category Correlation", maxHeight=None, maxWidth=None, nameLength=50)
	if outPath is not None:
		collected.writeFile(outPath)


def printQuestionToQuestionInSameCategoryCorrelation(responses, selected, categoriesByQName, outPath):
	collected = None
	for category, (qName1,qName2) in selected.items():
		q1 = responses.copyFeatures(qName1)
		q2 = responses.copyFeatures(qName2)

		q1 = -q1 if categoriesByQName[qName1,1] == 'male' else q1
		q2 = -q2 if categoriesByQName[qName2,1] == 'male' else q2

		q1.appendFeatures(q2)
		qs = q1
		corr = qs.featureSimilarities("correlation")
		corr.setPointName(0, category)
		corr.setFeatureName(1, 'Q to Q in Same Cat Corr')
		corr = corr.view(0,0,1,1)			
		if collected is None:
			collected = UML.createData("Matrix", [], featureNames=['Q to Q in Same Cat Corr'])
		collected.appendPoints(corr)

#	collected = abs(collected)
	collected.show("Selected Question To Category Correlation", maxHeight=None, maxWidth=None, nameLength=50)
	if outPath is not None:
		collected.writeFile(outPath)



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
	for catU, (q1, q2) in picked.items():
		cat = catU.lower()
		if cat[3] == '-':
			cat = 'non ' + cat[4:]
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
#		opts['title'] = str(bw[0][cat]) + " | " + catU + " | " + str(bw[1][cat])
		opts['title'] = catU
		opts['xlabel'] = ""
		opts['showPoints'] = False
		opts['xLimits'] = (-10, 10)
		opts['yLimits'] = (0, .17)
		plotDualWithBlendFill(mSubscale, fSubscale, bw[0][cat], bw[1][cat], **opts)
#		plotDualWithBlendFill(mSubscale, fSubscale, None, None, **opts)
#		if num > 0:
#		assert False
		num += 1


if __name__ == '__main__':
	#sys.exit(0)
	TRAIN_NUMBER = 300

	PRINT_CORR_FULLCAT_TO_FULLCAT = False
	PRINT_CORR_Q_TO_Q = False
	PRINT_CORR_Q_TO_FULLCAT = False
	PRINT_CORR_FULLCAT_TO_GENDER = False
	OUTPUT_FULLCAT_CORR_AND_PVALS = False

	REMOVE_PROBLEMS = True

	PRINT_CORR_SELCAT_TO_SELCAT = False
	PRINT_CORR_SELQ_TO_SELQ = False
	PRINT_CORR_SELQ_TO_SELCAT = False

	VERIFY_FROMFILE_CATEGORYSCORES = False
	VERIFY_MEANS_ORDERING_OF_SUBSCORES = True
	VERIFY_BANDWIDTH_SELECTION_FEASIBLE = False

	OUTPUT_SEL_QS_PER_CAT = False
	OUTPUT_SUBSCORE_CAT_CORR = False

	ADD_NOISE_TO_DATA = True
	RUN_BANDWIDTH_TRIALS = False
	OUTPUT_PLOTS = True

	UML.registerCustomLearner('Custom', KDEProbability)

	import time
	print time.asctime(time.localtime())

	sourceDir = sys.argv[1]
	path_categories = os.path.join(sourceDir, "inData", "question_categories.csv")
	path_responses = os.path.join(sourceDir, "inData", "question_data.csv")
	#
	outpath_selectedCatCorr = os.path.join(sourceDir, "analysis", "selectedCategoryCorr.csv")
	outpath_selectedQCorr = os.path.join(sourceDir, "analysis", "selectedQuestionCorr.csv")
	outpath_selectedQsToCatCorr = os.path.join(sourceDir, "analysis", "selectedQuestionToCategoryCorr.csv")
	outpath_selectedQsInCatCorr = os.path.join(sourceDir, "analysis", "selectedQToQInCategoryCorr.csv")
	outpath_selectedCorrToGender = os.path.join(sourceDir, "analysis", "selectedCorrToGender.csv")
	#
	outPath_selected = os.path.join(sourceDir, 'inData', "question_selected.csv")
	outpath_selectedInCatQsCorr = os.path.join(sourceDir, "analysis", "inCat_QtoQ_corr.csv")
	outPath_subscore_corr = os.path.join(sourceDir, "analysis", "subscore_category_correlation.csv")
	outPath_fullcat_corr_pval = os.path.join(sourceDir, "analysis", "category_to_gender_correlation_and_pvals.csv")
	outDir_plots = os.path.join(sourceDir, "plots")

	# Load data from the provided paths
	categoriesByQName, namesByCategory = loadCategoryData(path_categories)
	responses, categoryScores = loadResponseData(path_responses)

	if PRINT_CORR_FULLCAT_TO_FULLCAT:
		pass
	if PRINT_CORR_Q_TO_Q:
		pass
	if PRINT_CORR_Q_TO_FULLCAT:
		pass
	if PRINT_CORR_FULLCAT_TO_GENDER:
		printCategoryCorrelationToGender(namesByCategory, categoriesByQName, responses)
	if OUTPUT_FULLCAT_CORR_AND_PVALS:
		outputFile_CategoryCorrelationWithGender(outPath_fullcat_corr_pval, namesByCategory, categoriesByQName, responses)
	if VERIFY_FROMFILE_CATEGORYSCORES:
		checkFromFileCatScores(categoryScores, namesByCategory, categoriesByQName, responses)

	# remove the categories we have determined do not significantly distinguish between
	# genders according to p-value, or are too closely related to another, more expressive
	# category.
	# Also remove those questions we have since deem to be poorly worded, or overlapping
	# too strongly with questions in other categories.
	if REMOVE_PROBLEMS:
		removeProblemCategories(categoriesByQName, namesByCategory, responses)
		removeProblemQuestions(categoriesByQName, namesByCategory, responses)

	# Split gender / response data for subscore selection training and visualziation
	testFraction = float(responses.pointCount - TRAIN_NUMBER) / responses.pointCount
	responseTrain, genderTrain, responseTest, genderTest = responses.trainAndTestSets(testFraction, "male0female1")
	selected = determineBestSubScores(namesByCategory, categoriesByQName, responseTrain, genderTrain)

	printSelectedCategoryCorrelationMatrix(responseTrain, selected, categoriesByQName, outpath_selectedCatCorr)
	printSelectedQuestionCorrelationMatrix(responseTrain, selected, outpath_selectedQCorr)
	printSelectedQuestionToSelectedCategoryCorrelation(responseTrain, selected, categoriesByQName, outpath_selectedQsToCatCorr)
	printQuestionToQuestionInSameCategoryCorrelation(responseTrain, selected, categoriesByQName, outpath_selectedQsInCatCorr)

	outputSelectedCategoryCorrelationToGender(responseTrain, genderTrain, selected, categoriesByQName, outpath_selectedCorrToGender)

#	responseTest.show('sel')  # maxHeight=None, maxWidth=None)
#	print selected
	for cat,(q1,q2) in selected.items():
		print cat + '\t' + q1
		print '\t' + q2

#	printSelectedQuestionCorrelationInCategory(responseTest, selected, outpath_selectedInCatQsCorr)
	sys.exit(0)
	if PRINT_CORR_SELCAT_TO_SELCAT:
		printSelectedCategoryCorrelationMatrix(responseTest, selected, categoriesByQName, outpath_selectedCatCorr)
	if PRINT_CORR_SELQ_TO_SELQ:
		printSelectedQuestionCorrelationMatrix(responseTest, selected, outpath_selectedQCorr)
	if PRINT_CORR_SELQ_TO_SELCAT:
		printSelectedQuestionToSelectedCategoryCorrelation(responseTest, selected, categoriesByQName, outpath_selectedQsToCatCorr)

#	sys.exit(0)

	if VERIFY_MEANS_ORDERING_OF_SUBSCORES:
		verifyGenderAvgerageOrdering(selected, responseTest, genderTest)
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
		mBw, fBw = fixedBandwidth_same(namesByCategory, .22, .22)
#		mBw, fBw = fixedBandwidth_handAdjusted()
#		mBw, fBw = fixedBandwidth_all()
#		mBw, fBw = fixedBandwidth_drop5percent()
#		mBw, fBw = fixedBandwidth_noisy_all()		

	if OUTPUT_PLOTS:
		generatePlots(selected, categoriesByQName, responseTest_noisy, genderTest, outDir_plots, (mBw, fBw))

	print time.asctime(time.localtime())



	# load and clean
	# raw analysis
	# selection trial
	# selected questions and category subscore analysis
	# final prepeartion - noise, outlier removal
	# bandwidth
	# plot generation

	
	pass  # EOF marker
