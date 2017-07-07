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
from UML.calculate import residuals
from UML.examples.gender.gender_visualization_bandwidth import verifyBandwidthSelectionWorks
from UML.examples.gender.gender_visualization_bandwidth import KDEProbability
from UML.examples.gender.gender_visualization_bandwidth import bandwidthTrials
from UML.examples.gender.gender_visualization_bandwidth import fixedBandwidth_same
from UML.examples.gender.gender_visualization_bandwidth import fixedBandwidth_handAdjusted
from UML.examples.gender.gender_visualization_bandwidth import fixedBandwidth_all
from UML.examples.gender.gender_visualization_bandwidth import fixedBandwidth_drop5percent
from UML.examples.gender.gender_visualization_bandwidth import fixedBandwidth_noisy_all
from UML.examples.gender.gender_visualization_bandwidth import collateBW

#from UML.examples.gender.gender_visualization_bandwidth import *



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

#	categories.show("after", maxWidth=None, maxHeight=None, maxColumnWidth=33)

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
	"non resiliant to stress", "non image conscious", "optimistic", "talkative",
	and "power avoidant"

	"""
	# remove from categoriesByQName, a UML object where the category is the 0th value
	# of each point
	beforePointCount = categoriesByQName.pointCount
	catToRemove = ["Annoyable", "Non-Resilient To Illness", "Non-Resilient To Stress", "Non-Image Conscious", "Optimistic", "Talkative", "Power Avoidant"]

	for cat in catToRemove:
		assert cat in categoriesByQName.featureView(0)

	def removeFunc(point):
		return point[0] in catToRemove

	categoriesByQName.extractPoints(removeFunc)
	afterPointCount = categoriesByQName.pointCount
	assert beforePointCount - afterPointCount == 28
	for cat in catToRemove:
		assert cat not in categoriesByQName.featureView(0)


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
	assert responsesPCBefore - responsesPCAfter == 28

	# Remove from namesByCategory, a dict mapping category names to lists of questions
	nbcBefore = len(namesByCategory)
	for toRem in catToRemove:
		del namesByCategory[toRem]
	nbcAfter = len(namesByCategory)
	assert nbcBefore - nbcAfter == 7


def removeProblemQuestions(categoriesByQName, namesByCategory, responses):
	"""
	Remove those questions we have since deem to be poorly worded, or overlapping
	too strongly with questions in other categories. Specifically:
	"I get overwhelmed by difficult challenges.", "I freeze up under pressure.",
	"I need to plan out what I'm going to say in high pressure situations.",
	"I believe that I am better than others.",
	"I would enjoy having multiple sexual partners if I were single.", and
	"I often experience intense emotions."

	"""
	qsToRemove = ["I get overwhelmed by difficult challenges.",
					"I freeze up under pressure.",
					"I need to plan out what I'm going to say in high pressure situations.",
					"I believe that I am better than others.",
					"I would enjoy having multiple sexual partners if I were single.",
					"I often experience intense emotions."]

	# remove from categoriesByQName, a UML object where questions are point names
	beforePointCount = categoriesByQName.pointCount
	categoriesByQName.extractPoints(qsToRemove)
	afterPointCount = categoriesByQName.pointCount
	assert beforePointCount - afterPointCount == 6


	# Remove from namesByCategory, a dict mapping category names to lists of questions
	removed = 0
	for cat, qs in namesByCategory.items():
		keep = []
		for q in qs:
			if q not in qsToRemove:
				keep.append(q)
		removed += len(qs) - len(keep)
		namesByCategory[cat] = keep

	assert removed == 6


	# Remove from responses, where each feature is scores for a particular question
	responsesPCBefore = responses.featureCount
	responses.extractFeatures(qsToRemove)
	responsesPCAfter = responses.featureCount
	assert responsesPCBefore - responsesPCAfter == 6


def mergeProblemCategories(categoriesByQName, namesByCategory, responses):
	"""
	Merge certain problematically overlapping categories together, so that
	the best of the union of their questions will be choosen. Specifically
	merge: "non-manipulative" with "altruistic". For that pair, we will
	retain the name "altruistic"

	"""
	# adjust categoriesByQName, a UML object where the category is the 0th value
	# of each point
	beforePointCount = categoriesByQName.pointCount
	catToMerge = [("Non-Manipulative", 'Altruistic')]

	def changeFunc(val):
		for pair in catToMerge:
			if val == pair[0]:
				return pair[1]
			else:
				return None

	categoriesByQName.transformEachElement(changeFunc, features=0, skipNoneReturnValues=True)
	afterPointCount = categoriesByQName.pointCount
	assert beforePointCount - afterPointCount == 0
	for pair in catToMerge:
		assert pair[0] not in categoriesByQName.featureView(0)


	# Adjust namesByCategory, a dict mapping category names to lists of questions
	nbcBefore = len(namesByCategory)
	for pair in catToMerge:
		namesByCategory[pair[1]] += namesByCategory[pair[0]]
		del namesByCategory[pair[0]]
	nbcAfter = len(namesByCategory)
	assert nbcBefore - nbcAfter == 1


def renameResultantCategories(categoriesByQName, namesByCategory, responses, selected):
	"""
	Rename the categories, using prior knowledge for how the question selection
	will turn out, so that the chosen questions and category names correctly
	match

	"""
	# adjust categoriesByQName, a UML object where the category is the 0th value
	# of each point
	beforePointCount = categoriesByQName.pointCount
	# Old:New
	rename = {'Non-Eloquent':'Improvisational',
				'Risk Avoidant':'Risk Averse',
				'Altruistic':'Unselfish',
				'Worried':'Unworried',
				'Complexity Avoidant':'Complexity Seeking',
				'Agreeable':'Amicable',
				'Ordinary':'Unusual',
				'Non-Sexual':'Sex Focused',
				'Thin Skinned':"Thick Skinned"}

	def changeFunc(val):
		if val in rename:
			return rename[val]
		else:
			return None

	categoriesByQName.transformEachElement(changeFunc, features=0, skipNoneReturnValues=True)
	afterPointCount = categoriesByQName.pointCount
	assert beforePointCount - afterPointCount == 0
	for oldName in rename.keys():
		assert oldName not in categoriesByQName.featureView(0)

	# Adjust namesByCategory, a dict mapping category names to lists of questions
	nbcBefore = len(namesByCategory)
	for oldName in rename.keys():
		namesByCategory[rename[oldName]] = namesByCategory[oldName]
		del namesByCategory[oldName]
	nbcAfter = len(namesByCategory)
	assert nbcBefore - nbcAfter == 0
	for oldName in rename.keys():
		assert oldName not in namesByCategory

	# Adjust selected, a dict mapping category names to tuples of selected questions
	if selected is not None:
		selBefore = len(selected)
		for oldName in rename.keys():
			selected[rename[oldName]] = selected[oldName]
			del selected[oldName]
		selAfter = len(selected)
		assert selBefore - selAfter == 0
		for oldName in rename.keys():
			assert oldName not in selected


def determineBestSubScores(namesByCategory, categoriesByQName, responses, genderValue,
			forcedSelections):
	"""
	Out of the four questions for each category, pick those two that have the subscore
	most correlated with gender. These two questions are recorded in a dictionary,
	mapping category name to a tuple of question names.

	"""
	PRINT_CLOSE_CSV = False
	picked = {}

	for category, qs in namesByCategory.items():
		if category in forcedSelections:
			picked[category] = forcedSelections[category]
			continue

		gender = [categoriesByQName[q,1] for q in qs]

		pairwiseScale = []
		mapping = []
		qToQ = []
		for i,q in enumerate(qs):
			for j,u in enumerate(qs[i+1:]):
				curr = generateSubScale(responses, q, gender[i], u, gender[j+i+1])
				pairwiseScale.append(curr)
				mapping.append((q,u))

				qDat = responses.copyFeatures(q)
				uDat = responses.copyFeatures(u)
				qDat.appendFeatures(uDat)
				corr = qDat.featureSimilarities("correlation")
				qToQ.append(abs(corr[0,1]))

		scoreCorrGenPartial = functools.partial(scoreToGenderCorrelation, genders=genderValue)
		allCorr = map(scoreCorrGenPartial, pairwiseScale)
		best = max(allCorr)
		close = best*.1

		if PRINT_CLOSE_CSV:
			curr = []
			for i in xrange(len(allCorr)):
				if best - allCorr[i] < close:
					line = ','.join([category, mapping[i][0], mapping[i][1], str(qToQ[i]), str(allCorr[i])])
					curr.append(line)
			if len(curr) > 1:
				for val in curr:
					print val

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

def outputFile_SelectedQsMetadata(outPath, categoriesByQName, picked):
	allCats = categoriesByQName.featureView(0).copyAs("numpyarray")
	allCats = numpy.unique(allCats).tolist()
	raw = []
	for point in categoriesByQName.pointIterator():
		question = point.getPointName(0)
		category = point[0]

		if question not in picked[category]:
			continue
		
		newPoint = [question, category, point[1]]
		
		# do that inclusion/scale scoring of each possible category
		for possibleCat in allCats:
			if category == possibleCat:
				val = -1 if point[1] == 'male' else 1
			else:
				val = 0
#			newPoint.append(val)

		raw.append(newPoint)

#	fnames = ["Question", "Category", "Agreement Gender"] + allCats
	fnames = ["Question", "Category", "Agreement Gender"]
	toOutput = UML.createData("List", raw, featureNames=fnames)
#	print sum(ret.featureView(1))
#	toOutput.show("", maxWidth=120)

	toOutput.writeFile(outPath)


def outputFile_SelectedCatsMetadata(outPath, categoriesByQName, picked, responses, genders, scaleType):
	def extractFemale(point):
		pID = responses.getPointIndex(point.getPointName(0))
		return genders[pID] == 1

	toSplit = responses.copy()
	fResponses = toSplit.extractPoints(extractFemale)
	mResponses = toSplit

	raw = []
	for category, (q0, q1) in selected.items():
		allScore = generateSubScale(responses, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1], scaleType[category])
		
		mScore = generateSubScale(mResponses, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1], scaleType[category])
		mAvg = mScore.featureStatistics("mean")[0,0]
		
		fScore = generateSubScale(fResponses, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1], scaleType[category])
		fAvg = fScore.featureStatistics("mean")[0,0]

		corr = scoreToGenderCorrelation(allScore, genders)
		rawPoint = [category, scaleType[category], mAvg, fAvg, corr]
		raw.append(rawPoint)

	fnames = ["Category", "Agreement Gender", "Avg Male", "Avg Female", "Correlation to Gender"]
	toOutput = UML.createData("List", raw, featureNames=fnames)
#	print sum(ret.featureView(1))
#	toOutput.show("", maxWidth=120)

	toOutput.writeFile(outPath)



def outputFile_FullCategoryCorrelationWithGender(outPath, namesByCategory,
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


def printFullCategoryCorrelationToGender(namesByCategory, categoriesByQName, responses):
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


def printSelectedCategoryCorrelationMatrix(responses, gender, selected, categoriesByQName, partialCorr, outFile=None, scaleType=None):
	collected = None
	for category, (q0, q1) in selected.items():
		scale = scaleType[category] if scaleType is not None else 'female'
		sub = generateSubScale(responses, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1], scale)
		sub.setFeatureName(0, category)
		if collected is None:
			collected = sub
		else:
			collected.appendFeatures(sub)
	
	if partialCorr:
		residuals_collected = residuals(collected, gender)
		corrs = residuals_collected.featureSimilarities('correlation')
	else:
		corrs = collected.featureSimilarities('correlation')
	corrs.setPointName("Empathetic", "Compassionate")
	corrs.setFeatureName("Empathetic", "Compassionate")

	corrs.sortPoints(sortHelper=lambda x: x.getPointName(0))
	corrs.sortFeatures(sortHelper=lambda x: x.getFeatureName(0))
#	corrs.show("Selected Category Correlation Matrix", maxHeight=None, maxWidth=None, maxColumnWidth=25)
	if outFile is not None:
		corrs.writeFile(outFile)


def printSelectedCategoryPartialCorrelationGenderDiff(responses, gender, selected, categoriesByQName, outFileDiff, outFileBase, outFileSign, scaleType):
	collectedM = None
	collectedF = None

	resposnsesSafe = responses.copy()
	resposnsesSafe.appendFeatures(gender)
	males = resposnsesSafe.extractPoints(lambda x: x['male0female1'] == 0)
	females = resposnsesSafe

	males.extractFeatures('male0female1')
	females.extractFeatures('male0female1')

	for category, (q0, q1) in selected.items():
		scale = scaleType[category] if scaleType is not None else 'female'
		subM = generateSubScale(males, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1], scale)
		subF = generateSubScale(females, q0, categoriesByQName[q0,1], q1, categoriesByQName[q1,1], scale)
		subM.setFeatureName(0, category)
		subF.setFeatureName(0, category)
		if collectedM is None:
			collectedM = subM
		else:
			collectedM.appendFeatures(subM)

		if collectedF is None:
			collectedF = subF
		else:
			collectedF.appendFeatures(subF)

	corrsM = collectedM.featureSimilarities('correlation')
	corrsF = collectedF.featureSimilarities('correlation')

	corrsM.setPointName("Empathetic", "Compassionate")
	corrsM.setFeatureName("Empathetic", "Compassionate")
	corrsF.setPointName("Empathetic", "Compassionate")
	corrsF.setFeatureName("Empathetic", "Compassionate")

	corrsM.sortPoints(sortHelper=lambda x: x.getPointName(0))
	corrsM.sortFeatures(sortHelper=lambda x: x.getFeatureName(0))
	corrsF.sortPoints(sortHelper=lambda x: x.getPointName(0))
	corrsF.sortFeatures(sortHelper=lambda x: x.getFeatureName(0))

	avgCorr = (corrsM + corrsF) / 2.0
	basePartialCorr = UML.createData("Matrix", outFileBase)

	signMatrix = numpy.empty((avgCorr.pointCount, avgCorr.featureCount), dtype=int)
	for i in range(signMatrix.shape[0]):
		for j in range(signMatrix.shape[1]):
			if abs(avgCorr[i,j]) <= 0.02 and abs(basePartialCorr[i,j]) <= 0.02:
				signMatrix[i,j] = 0
			elif avgCorr[i,j] > 0 and basePartialCorr[i,j] > 0:
				signMatrix[i,j] = 1
			elif avgCorr[i,j] < 0 and basePartialCorr[i,j] < 0:
				signMatrix[i,j] = 1
			elif avgCorr[i,j] < 0 and basePartialCorr[i,j] > 0:
				signMatrix[i,j] = -1
			elif avgCorr[i,j] > 0 and basePartialCorr[i,j] < 0:
				signMatrix[i,j] = -1
			else:
				raise RuntimeError("ALARM")

	signObj = UML.createData("List", signMatrix, pointNames=avgCorr.getPointNames(), featureNames=avgCorr.getFeatureNames())
	signObj.writeFile(outFileSign)

	diff = avgCorr - basePartialCorr

	diff.writeFile(outFileDiff)


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
#	collected.show("Selected Question To Category Correlation", maxHeight=None, maxWidth=None, nameLength=50)
	if outPath is not None:
		collected.writeFile(outPath)


def outputFile_selected_data(responses, categoriesByQName, selected, outPath):
	toUse = responses.copy()
	gender = toUse.extractFeatures("male0female1")
	selectedQs = toUse
	selectedNames = []

	for fname in selectedQs.getFeatureNames():
		currCat = categoriesByQName[fname,0]
		if fname in selected[currCat]:
			selectedNames.append(fname)

	selectedQs = selectedQs.extractFeatures(selectedNames)
	selectedQs.appendFeatures(gender)
	selectedData = selectedQs

	if outPath is not None:
		selectedData.writeFile(outPath)


def outputFile_selected_and_transformed_data(responses, categoriesByQName, scaleType, selected, outPath):
	toUse = responses.copy()
	gender = toUse.extractFeatures("male0female1")
	responsesOnly = toUse
	rescale = []
	selectedNames = []

	for fname in responsesOnly.getFeatureNames():
		currCat = categoriesByQName[fname,0]
		if fname in selected[currCat]:
			selectedNames.append(fname)

			if scaleType[currCat] == 'female':
				scaleMod = 1 if categoriesByQName[fname,1] == 'female' else -1
			else:
				scaleMod = -1 if categoriesByQName[fname,1] == 'female' else 1
			rescale.append(scaleMod)

	responsesOnly = responsesOnly.extractFeatures(selectedNames)

#	responsesOnly.pointView(0).show("beforeNorm", maxWidth=None)

	rescaleObj = UML.createData("Matrix", rescale)
	responsesOnly.normalizeFeatures(divide=rescaleObj)

#	responsesOnly.pointView(0).show("AfterNorm", maxWidth=None)

	responsesOnly.appendFeatures(gender)
	transformedData = responsesOnly

	if outPath is not None:
		transformedData.writeFile(outPath)


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


def generateSubScale(data, qA_ID, qA_Gender, qB_ID, qB_Gender, scale_Gender='female'):
	qA = data.copyFeatures(qA_ID)
	qB = data.copyFeatures(qB_ID)
	qA.setFeatureNames(None)
	qB.setFeatureNames(None)

	if scale_Gender == 'female':
		qA = -qA if qA_Gender == 'male' else qA
		qB = -qB if qB_Gender == 'male' else qB
	else:
		qA = -qA if qA_Gender == 'female' else qA
		qB = -qB if qB_Gender == 'female' else qB

	return (qA + qB) / 2.0

def setupCategoryScaleTypes(categoriesByQName, selected, includeMale):
	if includeMale:
		ret = {'Unworried':'male', 'Risk Averse':'female',
				'Unselfish':'female', 'Emotionally Aware':'female',
				'Unusual':'male', 'Warm':'female', 'Amicable':'female',
				'Improvisational':'male', 'Thick Skinned':'male',
				'Forgiving':'female', 'Sex Focused':'male',
				'Complexity Seeking':'male', 'Empathetic':'female'}

		if selected is not None:
			for cat in ret:
				assert cat in selected
	else:
		ret = {}
		for cat in selected.keys():
			ret[cat] = 'female'

	def unpack(point):
		return ret[point[0]]

	catScaleFeature = categoriesByQName.calculateForEachPoint(unpack)
	catScaleFeature.setFeatureName(0, 'genderHigherAvgOfCat')

	if categoriesByQName.featureCount == 3:
		categoriesByQName.extractFeatures('genderHigherAvgOfCat')
	categoriesByQName.appendFeatures(catScaleFeature)

	return ret


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


def generatePlots(picked, categoriesByQName, responses, genderValue, outDir, bw, scaleType):
	def extractFemale(point):
		pID = responses.getPointIndex(point.getPointName(0))
		return genderValue[pID] == 1

	toSplit = responses.copy()
	femalePoints = toSplit.extractPoints(extractFemale)
	malePoints = toSplit

	num = 0
	for catU, (q1, q2) in picked.items():
		catScaleGender = scaleType[catU]
		# need this to interface with legacy saved bandwidth selections
		cat = catU.lower()
		if cat[3] == '-':
			cat = 'non ' + cat[4:]

		if catU in bw[0]:
			bw[0][cat] = bw[0][catU]
		if catU in bw[1]:
			bw[1][cat] = bw[1][catU]

		q1Gender = categoriesByQName[q1,1]
		q2Gender = categoriesByQName[q2,1]

		fSubscale = generateSubScale(femalePoints, q1, q1Gender, q2, q2Gender, catScaleGender)  # .extractPoints(end=20)
		mSubscale = generateSubScale(malePoints, q1, q1Gender, q2, q2Gender, catScaleGender)  # .extractPoints(end=20)

#		fSubscale.show("F", maxWidth=None, maxHeight=None)

#		kdeAndHistogramPlot(mSubscale, title=cat+" Ma", show=True)
#		kdeAndHistogramPlot(fSubscale, title=cat+" Fem", show=True)
		fileName = os.path.join(outDir, cat)
		opts = {}
		opts['fileName'] = fileName
#		opts['fileName'] = None
		opts['show'] = False
#		opts['title'] = str(bw[0][cat]) + " | " + catU + " | " + str(bw[1][cat])
		opts['title'] = catU
#		opts['xlabel'] = "Score"
		opts['ylabel'] = "Frequency of Score"
		opts['showPoints'] = False
		opts['xLimits'] = (-10, 10)
		opts['yLimits'] = (0, .175)
		opts['legendContents'] = ("Males", "Females")
		opts['normalizeAUC'] = True
		plotDualWithBlendFill(mSubscale, fSubscale, bw[0][cat], bw[1][cat], **opts)
#		plotDualWithBlendFill(mSubscale, fSubscale, None, None, **opts)
#		if num > 0:
#		assert False
		num += 1


if __name__ == '__main__':
	#sys.exit(0)
	# Constants controlling how the data is split in train and test sets
	TRAIN_NUMBER = 300
	SPLITSEED = 42

	# Flags enabling analysis of the full, orignal category data
	PRINT_CORR_FULLCAT_TO_GENDER = False
	OUTPUT_FULLCAT_CORR_AND_PVALS = True

	# Flags enabling analysis of the training set category selection
	OUTPUT_CORR_SELCAT_TO_SELCAT = True
	OUTPUT_CORR_SELQ_TO_SELQ = True
	OUTPUT_CORR_SELQ_TO_SELCAT = False
	OUTPUT_CORR_SELCAT_TO_GENDER = True
	OUTPUT_INCAT_Q_TO_Q = True

	# Flags enabling results reporting of category selection
	PRINT_SELQS = False
	OUTPUT_SEL_QS_PER_CAT = True
	OUTPUT_CORR_SELCAT_TO_SELCAT_OUTOFSAMPLE = False
	OUTPUT_PARTIALCORR_SELCAT_TO_SELCAT_OUTOFSAMPLE = True


	# Flags enabling confirmation of our data and processes
	VERIFY_FROMFILE_CATEGORYSCORES = False
	VERIFY_MEANS_ORDERING_OF_SUBSCORES = True
	VERIFY_BANDWIDTH_SELECTION_FEASIBLE = False

	# Flags enabling modifications to the processing pipeline
	CATEGORY_CLEANUP = True
	RENAME_CATEGORIES_INCLUDE_MALE_SCALES_BEFORE_ANALYSIS = False
	RENAME_CATEGORIES_INCLUDE_MALE_SCALES_AFTER_ANALYSIS = True
#	SOME_MALE_SCALES = RENAME_CATEGORIES_INCLUDE_MALE_SCALES_BEFORE_ANALYSIS or RENAME_CATEGORIES_INCLUDE_MALE_SCALES_AFTER_ANALYSIS
	ADD_NOISE_TO_DATA = False
	RUN_BANDWIDTH_TRIALS = False
	SAVE_BANDWIDTH_TRIAL_RESULTS = True
	OUTPUT_PLOTS = False

	# Flag for outputing transformed data ala what we would expect from the next trial
	OUTPUT_DATA_SELECTED_AND_TRANSFORMED_SCALE = True

#	UML.setRandomSeed(54324)

	UML.registerCustomLearner('Custom', KDEProbability)

	import time
	print time.asctime(time.localtime())

	# Source Data
	sourceDir = sys.argv[1]
	path_categories = os.path.join(sourceDir, "inData", "question_categories.csv")
	path_responses = os.path.join(sourceDir, "inData", "question_data.csv")

	# Output location for transformed data
	outpath_responses_selected_and_transformed = os.path.join(sourceDir, "inData", "questions_selected_and_formated.csv")
	outpath_responses_selected_only = os.path.join(sourceDir, "inData", "questions_selected.csv")

	# Output files for full category score analysis
	outPath_fullcat_corr_pval = os.path.join(sourceDir, "analysis", "category_to_gender_correlation_and_pvals.csv")
	
	# Output files for category selection analysis and results
	outpath_selected_CatCorr = os.path.join(sourceDir, "analysis", "selected_CategoryCorr.csv")
	outpath_selected_CatCorr_outOfSample = os.path.join(sourceDir, "analysis", "selected_CategoryCorr_test.csv")
	outPath_selected_Cat_PartialCorr_outSample = os.path.join(sourceDir, "analysis", "selected_CategoryPartialCorr_test.csv")
	outPath_selected_Cat_PartialCorrDiff_outSample = os.path.join(sourceDir, "analysis", "selected_CategoryPartialCorrDiff_test.csv")
	outPath_selected_Cat_PartialCorrSign_outSample = os.path.join(sourceDir, "analysis", "selected_CategoryPartialCorrSign_test.csv")
	outpath_selected_QCorr = os.path.join(sourceDir, "analysis", "selected_QuestionCorr.csv")
	outpath_selected_QsToCatCorr = os.path.join(sourceDir, "analysis", "selected_QuestionToCategoryCorr.csv")
	outpath_selected_QsInCatCorr = os.path.join(sourceDir, "analysis", "selected_QToQInCategoryCorr.csv")
	outpath_selected_CorrToGender = os.path.join(sourceDir, "analysis", "selected_CatToGenderCorr.csv")
	outPath_selected_results = os.path.join(sourceDir, 'inData', "questions_selected.csv")
	outPath_selectedQ_metadata = os.path.join(sourceDir, 'inData', "questions_selectedMetadata.csv")
	outPath_selectedCat_metadata = os.path.join(sourceDir, 'inData', "categories_selectedCatMetadata.csv")

	# Output directory for bandwidth trial resutlts
	outDir_BW_results = os.path.join(sourceDir, "analysis", "BW")
	outDir_BW_results = outDir_BW_results if SAVE_BANDWIDTH_TRIAL_RESULTS else None

	# Output files for visualizations
	outDir_plots = os.path.join(sourceDir, "plots")


	# Load data from the provided paths
	categoriesByQName, namesByCategory = loadCategoryData(path_categories)
	responses, categoryScores = loadResponseData(path_responses)

	# Confirm that our calculated full category scores match those previously calculated
	# and already present in the responses data.
	if VERIFY_FROMFILE_CATEGORYSCORES:
		checkFromFileCatScores(categoryScores, namesByCategory, categoriesByQName, responses)

	# Optional analysis of the full, original categories
	if PRINT_CORR_FULLCAT_TO_GENDER:
		printFullCategoryCorrelationToGender(namesByCategory, categoriesByQName, responses)
	if OUTPUT_FULLCAT_CORR_AND_PVALS:
		outputFile_FullCategoryCorrelationWithGender(outPath_fullcat_corr_pval, namesByCategory, categoriesByQName, responses)

	# Adjust the categories, questions, and names given the results of our data
	# and selection analysis.
	# Removes the categories we have determined do not significantly distinguish between
	# genders according to p-value, or are a special case of a more expressive
	# category.
	# Removes those questions we have since deemed to be poorly worded, or overlapping
	# too strongly with questions in other categories.
	# Force certain questions to be selected by removing all other options
	# Combines categories we have determined are really asking about the same thing
	# Rename the resultant categories given how we know the selection process
	# will turn out.
	if CATEGORY_CLEANUP:
		removeProblemCategories(categoriesByQName, namesByCategory, responses)
		removeProblemQuestions(categoriesByQName, namesByCategory, responses)
		mergeProblemCategories(categoriesByQName, namesByCategory, responses)
		forcedSelections = {'Altruistic':
								('I am out for my own personal gain.',
								'I look out for myself first before I look out for others.'),
							'Unselfish':
								('I am out for my own personal gain.',
								'I look out for myself first before I look out for others.')}
	else:
		forcedSelections = {}

	# Finalize the category names, which has the effect of changing the correct
	# scale for questions from always average female agreement, to male / female
	# average agree per different questions - which may make the selection analysis
	# results look strange, since they are all keyed to female scales only.
	if RENAME_CATEGORIES_INCLUDE_MALE_SCALES_BEFORE_ANALYSIS:
		renameResultantCategories(categoriesByQName, namesByCategory, responses, None)
#		scaleType = setupCategoryScaleTypes(categoriesByQName, None, True)
#		scaleType = setupCategoryScaleTypes(categoriesByQName, None, False)
#	else:
#		scaleType = setupCategoryScaleTypes(categoriesByQName, None, False)

	# Split gender / response data for subscore selection training and visualziation
	testFraction = float(responses.pointCount - TRAIN_NUMBER) / responses.pointCount
	UML.setRandomSeed(SPLITSEED)
	responseTrain, genderTrain, responseTest, genderTest = responses.trainAndTestSets(testFraction, "male0female1", randomOrder=True)
	selected = determineBestSubScores(namesByCategory, categoriesByQName, responseTrain, genderTrain, forcedSelections)

	# Verify our split and selection process correctness
	if VERIFY_MEANS_ORDERING_OF_SUBSCORES:
		verifyGenderAvgerageOrdering(selected, responseTest, genderTest)
	if VERIFY_BANDWIDTH_SELECTION_FEASIBLE:
		verifyBandwidthSelectionWorks(responseTest, genderTest)

	# Analysis of category selection using the training data
	if OUTPUT_CORR_SELCAT_TO_SELCAT:
		printSelectedCategoryCorrelationMatrix(responseTrain, genderTrain, selected, categoriesByQName, False, outpath_selected_CatCorr)
	if OUTPUT_CORR_SELQ_TO_SELQ:
		printSelectedQuestionCorrelationMatrix(responseTrain, selected, outpath_selected_QCorr)
	if OUTPUT_CORR_SELQ_TO_SELCAT:
		printSelectedQuestionToSelectedCategoryCorrelation(responseTrain, selected, categoriesByQName, outpath_selected_QsToCatCorr)
	if OUTPUT_INCAT_Q_TO_Q:
		printQuestionToQuestionInSameCategoryCorrelation(responseTrain, selected, categoriesByQName, outpath_selected_QsInCatCorr)
	if OUTPUT_CORR_SELCAT_TO_GENDER:
		outputSelectedCategoryCorrelationToGender(responseTrain, genderTrain, selected, categoriesByQName, outpath_selected_CorrToGender)

	# Finalize the category names, which has the effect of changing the correct
	# scale for questions from always average female agreement, to male / female
	# average agree per different questions
	if RENAME_CATEGORIES_INCLUDE_MALE_SCALES_AFTER_ANALYSIS:
		renameResultantCategories(categoriesByQName, namesByCategory, responses, selected)
		# If we have renamed categories, then some categories now have names which
		# imply a higher male average score. We record these to be used later in 
		# subscale generateion.
		scaleType = setupCategoryScaleTypes(categoriesByQName, selected, True)
	else:
		scaleType = setupCategoryScaleTypes(categoriesByQName, selected, False)

	# Report results of category selection
	if OUTPUT_SEL_QS_PER_CAT:
		outputFile_SelectedQsPerCategory(outPath_selected_results, categoriesByQName, selected)
		outputFile_SelectedQsMetadata(outPath_selectedQ_metadata, categoriesByQName, selected)
		outputFile_SelectedCatsMetadata(outPath_selectedCat_metadata, categoriesByQName, selected, responseTest, genderTest, scaleType)
	if PRINT_SELQS:
		for cat,(q1,q2) in selected.items():
			print cat + '\t' + q1
			print '\t' + q2

	if OUTPUT_CORR_SELCAT_TO_SELCAT_OUTOFSAMPLE:
		printSelectedCategoryCorrelationMatrix(responseTest, genderTest, selected, categoriesByQName, False, outpath_selected_CatCorr_outOfSample, scaleType)
	if OUTPUT_PARTIALCORR_SELCAT_TO_SELCAT_OUTOFSAMPLE:
		printSelectedCategoryCorrelationMatrix(responseTrain, genderTrain, selected, categoriesByQName, True, outPath_selected_Cat_PartialCorr_outSample, scaleType)
		printSelectedCategoryPartialCorrelationGenderDiff(responseTrain, genderTrain, selected, categoriesByQName, outPath_selected_Cat_PartialCorrDiff_outSample, outPath_selected_Cat_PartialCorr_outSample, outPath_selected_Cat_PartialCorrSign_outSample, scaleType)

	if OUTPUT_DATA_SELECTED_AND_TRANSFORMED_SCALE:
		outputFile_selected_data(responses, categoriesByQName, selected, outpath_responses_selected_only)
		outputFile_selected_and_transformed_data(responses, categoriesByQName, scaleType, selected, outpath_responses_selected_and_transformed)

	sys.exit()

	# responses were only to one digit, which makes them clump for bandwidth trials.
	# add some noise to loosen things up without going outside the range of what would
	# round to the given value
	if ADD_NOISE_TO_DATA:
		responseTest_noisy = addNoiseToResponses(responseTest)
	else:
		responseTest_noisy = responseTest
	
	if RUN_BANDWIDTH_TRIALS:
		mBw, fBw = bandwidthTrials(selected, categoriesByQName, responseTest_noisy, genderTest, scaleType, True, outDir_BW_results)
	else:
		mBw, fBw = fixedBandwidth_same(namesByCategory, .25, .25)
#		mBw, fBw = fixedBandwidth_handAdjusted()
#		mBw, fBw = fixedBandwidth_all()
#		mBw, fBw = fixedBandwidth_drop5percent()
#		mBw, fBw = fixedBandwidth_noisy_all()		

#	avg = 0
#	for k in mBw.keys():
#		avg += mBw[k] + fBw[k]
#	print avg / float(len(mBw)*2)

#	collateBW()

	if OUTPUT_PLOTS:
		generatePlots(selected, categoriesByQName, responseTest_noisy, genderTest, outDir_plots, (mBw, fBw), scaleType)

	print time.asctime(time.localtime())



	# load and clean
	# raw analysis
	# selection trial
	# selected questions and category subscore analysis
	# final prepeartion - noise, outlier removal
	# bandwidth
	# plot generation

	
	pass  # EOF marker
