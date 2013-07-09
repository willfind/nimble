"""
The Stopwatch class is used for timing various tasks (training classifiers,
testing classifiers)within the UML code.  Can time any generic task based
on task name.
"""

import time
import nose
from nose.tools import *
from UML.exceptions import MissingEntryException
from UML.exceptions import ImproperActionException

class Stopwatch(object):

	def __init__(self):
		self.startTimes = dict()
		self.stopTimes = dict()
		self.cumulativeTimes = dict()
		self.isRunningStatus = dict()

	def start(self, taskName):
		"""
		Record the start time for the provided taskName.  If there is already
		a start time associated with taskName, overwrite the old start time.
		TODO: May need to change that to raising an exception, instead of overwriting).
		"""
		if taskName in self.isRunningStatus and self.isRunningStatus[taskName] == True:
			raise ImproperActionException("Task: " + taskName + " has already been started.")
		else:
			self.startTimes[taskName] = time.clock()
			if taskName not in self.cumulativeTimes:
				self.cumulativeTimes[taskName] = 0.0
			self.isRunningStatus[taskName] = True

	def stop(self, taskName):
		"""
			Record the stop time for the provided taskName.  If there is already a stop time
			associated with taskName, overwrite the old stop time.
			TODO: Possibly raise an exception instead of overwriting existing stop time, if there
			is already an entry for taskName.
		"""
		if taskName not in self.startTimes or taskName not in self.isRunningStatus:
			raise MissingEntryException([taskName], "Tried to stop task that was not started in Stopwatch.stop()")
		elif not self.isRunningStatus[taskName]:
			raise ImproperActionException("Unable to stop task that has already stopped")

		self.stopTimes[taskName] = time.clock()
		self.isRunningStatus[taskName] = False
		if taskName in self.cumulativeTimes:
			self.cumulativeTimes[taskName] += self.stopTimes[taskName] - self.startTimes[taskName]
		else:
			self.cumulativeTimes[taskName] = self.stopTimes[taskName] - self.startTimes[taskName]

	def reset(self, taskName, resetCumulativeTime=True):
		"""
			Reset/remove all values in stopwatch associated with taskName.  If resetCumulativeTime is
			True, get rid of the running total of time for taskName.  If resetCumulativeTime is false,
			the running total is preserved but taskName will be stopped and isRunning(taskName) will
			return False.
		"""
		if taskName in self.startTimes:
			del self.startTimes[taskName]

		if taskName in self.stopTimes:
			del self.stopTimes[taskName]

		if taskName in self.isRunningStatus:
			self.isRunningStatus[taskName] = False

		if resetCumulativeTime:
			self.cumulativeTimes[taskName] = 0.0

	def isRunning(self, taskName):
		"""
			Check if the stopwatch is currently timing a task with taskName (i.e. the timer has been
			started but not stopped for said task).  Returns true if start() has been called for
			this taskName but neither stop() nor reset() have been called for this taskName.  Otherwise
			returns false.
		"""
		if taskName in self.isRunningStatus:
			if self.isRunningStatus[taskName] == True:
				return True
		else:
			return False

	def calcRunTime(self, taskName):
		"""
			Calculate the time it took for the task associated with taskName to complete,
			based on the stored start time and stop time.  If the timer is still timing the
			task associated with taskName (i.e. isRunning(taskName) is True), raises an
			ImproperActionException.
		"""
		if taskName not in self.cumulativeTimes or taskName not in self.isRunningStatus:
			raise MissingEntryException([taskName], "Missing entry when trying to calculate total task run time: " + str(taskName))
		elif self.isRunningStatus[taskName] == True:
			raise ImproperActionException('Can\'t calculate total running time for ' + taskName + ', as it is still running')
		else:
			return self.cumulativeTimes[taskName]



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
