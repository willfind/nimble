"""
Tests for the functions defined in UML.calculate.utility

"""

import numpy
from nose.tools import *

import UML
from UML.calculate import cosineSimilarity
from UML.calculate import detectBestResult
from UML.calculate import fractionCorrect
from UML.calculate import fractionIncorrect
from UML.calculate import meanAbsoluteError
from UML.calculate import rootMeanSquareError
from UML.calculate import rSquared
from UML.calculate import varianceFractionRemaining

from UML.exceptions import ArgumentException

####################
# detectBestResult #
####################

# labels, best, all
# inconsistent between different mixed runs

@raises(ArgumentException)
def test_detectBestResult_labels_inconsistentForDifferentKnowns():
	def foo(knowns, predicted):
		rawKnowns = knowns.copyAs("numpyarray")
		rawPred = predicted.copyAs("numpy.array")

		# case: knowns all zeros
		if not numpy.any(rawKnowns):
			# case: predictd all right
			if not numpy.any(rawPred):
				return 0
			# case: predicted all wrong
			elif numpy.all(rawPred):
				return 1
			else:
				return UML.calculate.meanAbsoluteError(knowns, predicted)
		# case: knowns all ones
		elif numpy.all(rawKnowns):
			# case: predicted all wrong
			if not numpy.any(rawPred):
				return 0
			# case: predictd all right
			elif numpy.all(rawPred):
				return 1
			else:
				return 1 - UML.calculate.meanAbsoluteError(knowns, predicted)

		return UML.calculate.meanAbsoluteError(knowns, predicted)

	detectBestResult(foo)

@raises(ArgumentException)
def test_detectBestResult_labels_allcorrect_equals_allwrong():
	detectBestResult(lambda x,y: 20)


@raises(ArgumentException)
def test_detectBestResult_labels_nonmonotonic_minmizer():
	def foo(knowns, predicted):
		ret = UML.calculate.fractionIncorrect(knowns, predicted)
		if ret > .25 and ret < .5:
			return .6
		if ret >= .5 and ret < .75:
			return .4
		return ret

	detectBestResult(foo)


@raises(ArgumentException)
def test_detectBestResult_labels_nonmonotonic_maxizer():
	def foo(knowns, predicted):
		if knowns == predicted:
			return 100
		else:
			return UML.randomness.pythonRandom.randint(0,20)

	detectBestResult(foo)


@raises(ArgumentException)
def test_detectBestResult_wrongSignature_low():
	def tooFew(arg1):
		return 0
	detectBestResult(tooFew)


@raises(ArgumentException)
def test_detectBestResult_wrongSignature_high():
	def tooMany(arg1, arg2, arg3):
		return 0
	detectBestResult(tooMany)


def test_detectBestResult_exceptionsAreReported():
	wanted = "SPECIAL TEXT"

	def neverWorks(knowns, predicted):
		raise ArgumentException(wanted)

	try:
		detectBestResult(neverWorks)
		assert False  # we expected an exception in this test
	except ArgumentException as ae:
		assert wanted in ae.value 


def _backend(performanceFunction, optimality):
	assert performanceFunction.optimal == optimality
	try:
		performanceFunction.optimal = None
		result = detectBestResult(performanceFunction)
		assert result == optimality
	finally:
		performanceFunction.optimal = optimality


def test_detectBestResult_rootMeanSquareError():
	_backend(rootMeanSquareError, 'min')


def test_detectBestResult_meanAbsoluteError():
	_backend(meanAbsoluteError, 'min')
	

def test_detectBestResult_fractionIncorrect():
	_backend(fractionIncorrect, 'min')


def test_detectBestResult_cosineSimilarity():
	_backend(cosineSimilarity, 'max')


def test_detectBestResult_rSquared():
	_backend(rSquared, 'max')


def test_detectBestResult_varianceFractionRemaining():
	_backend(varianceFractionRemaining, 'min')
