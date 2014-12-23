
from nose.tools import *

from UML.calculate import cosineSimilarity
from UML.calculate import detectBestResult
from UML.calculate import fractionIncorrect
from UML.calculate import fractionIncorrectBottom10
from UML.calculate import meanAbsoluteError
from UML.calculate import rootMeanSquareError




####################
# detectBestResult #
####################

def testDectection_rootMeanSquareError():
	result = detectBestResult(rootMeanSquareError)
	assert result == 'min'

def testDectection_meanAbsoluteError():
	result = detectBestResult(meanAbsoluteError)
	assert result == 'min'

def testDectection_fractionIncorrect():
	result = detectBestResult(fractionIncorrect)
	assert result == 'min'

def testDetection_cosineSimilarity():
	result = detectBestResult(cosineSimilarity)
	assert result == 'max'

def testDectection_fractionIncorrectBottom10():
	result = detectBestResult(fractionIncorrectBottom10)
	assert result == 'min'
