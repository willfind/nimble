"""
Learners and metrics useful for determining an appropriate bandwidth
for the Kernel Density Estimation of some sample data. Also includes
hard coded results from running on the 2nd round (84 question) gender
survery data.

"""

import numpy
import scipy
import sys
import matplotlib.pyplot as plt


KDE_HELPER_PATH = "/home/tpburns/Dropbox/ML_intern_tpb/python_workspace/kdePlotting"
#ORIG_HELPER_PATH = "/home/tpburns/Dropbox/ML_intern_tpb/python_workspace/"
sys.path.append(KDE_HELPER_PATH)
#sys.path.append(ORIG_HELPER_PATH)
#from fancyHistogram import kdeAndHistogramPlot
from multiplot import plotDualWithBlendFill

import UML
from UML.randomness import pythonRandom
from UML.randomness import numpyRandom
from UML.examples.gender.gender_visualization_bandwidth import *



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


#def fixedBandwidth_noisy_drop5percent():
#	mbw = {	}

#	fbw = {	}

#	return mbw,fbw

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
	names = {}
	for catName in namesByCategory.keys():
		adjusted = ' '.join(catName.lower().split('-'))
		names[adjusted] = None
	
	mbw = {}
	for k in names:
		mbw[k] = mVal

	fbw = {}
	for k in names:
		fbw[k] = mVal

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


def verifyBandwidthSelectionWorks(responses, genderValue):
	"""
	Generate plots of known distributions using several methods of bandwidth selection,
	comparing the resultant estimated distributions with the knowns in a plot.

	"""
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
	bw = tuple([bwBaseM * (1.1 ** i) for i in xrange(-15,15)])
	mfolds = 10
	mAll = UML.crossValidateReturnAll("custom.KDEProbability", genDataM, None, bandwidth=bw, numFolds=mfolds, performanceFunction=LogLikelihoodSum)
	mBW = cvUnpackBest(mAll, False)[0]['bandwidth']

	bwBaseF = silver_factor(genDataM.pointCount, genDataM.featureCount)
	bw = tuple([bwBaseF * (1.1 ** i) for i in xrange(-15,15)])
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
	bw = tuple([bwBaseM * (1.1 ** i) for i in xrange(-15,15)])
	mfolds = 10
	mAll = UML.crossValidateReturnAll("custom.KDEProbability", genDataM, None, bandwidth=bw, numFolds=mfolds, performanceFunction=LogLikelihoodSum)
	mBW = cvUnpackBest(mAll, False)[0]['bandwidth']

	bwBaseF = silver_factor(genDataM.pointCount, genDataM.featureCount)
	bw = tuple([bwBaseF * (1.1 ** i) for i in xrange(-15,15)])
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
