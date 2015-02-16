
from nose.tools import *

from UML.logger.stopwatch import Stopwatch
from UML.exceptions import ImproperActionException
from UML.exceptions import MissingEntryException


def testBasicFuncs():
	watch = Stopwatch()
	watch.start('test')
	# giberish, to eat some time
	for i in range(100000):
		x = (i * (i + i)) / 3.0
	watch.stop('test')
	runTime = watch.calcRunTime('test')
	assert runTime > 0.0
	assert runTime < 10.0

	watch = Stopwatch()
	watch.start('test')
	watch.stop('test')
	watch.startTimes['test'] = 0.00
	watch.stopTimes['test'] = 5.00
	watch.cumulativeTimes['test'] = 5.00

	runTime = watch.calcRunTime('test')
	assert runTime == 5.0
	assert not watch.isRunning('test')

	watch.start('test')
	assert watch.isRunning('test')

	watch.stop('test')
	watch.reset('test', False)
	assert not watch.isRunning('test')
	assert watch.calcRunTime('test') > 0.0
	watch.reset('test')
	assert watch.calcRunTime('test') == 0.0

@raises(ImproperActionException)
def testDoubleStart():
	watch = Stopwatch()
	watch.start('test')
	watch.start('test')

@raises(MissingEntryException)
def testStopMissingEntry():
	watch = Stopwatch()
	watch.start('test')
	watch.stop('uncle')

@raises(ImproperActionException)
def testCalcRunTimeImproperAction():
	watch = Stopwatch()
	watch.start('test')
	watch.calcRunTime('test')

@raises(MissingEntryException)
def testCalcRunTimeMissingEntry():
	watch = Stopwatch()
	watch.start('test')
	watch.calcRunTime('uncle')

@raises(ImproperActionException)
def testImproperActionStop():
	watch = Stopwatch()
	watch.start('test')
	watch.stop('test')
	watch.stop('test')
