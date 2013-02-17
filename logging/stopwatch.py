"""
	The Stopwatch class is used for timing various tasks (training classifiers,
	testing classifiers)within the UML code.  Can time any generic task based
	on task name.
"""

import time
from .utility.custom_exceptions import MissingEntryException

class Stopwatch(object):

	def __init__(self):
		self.startTimes = dict()
		self.stopTimes = dict()
		self.cumulativeTimes = dict()
		self.isRunning = dict()

	def start(self, taskName):
		"""
			Record the start time for the provided taskName.  If there is already
			a start time associated with taskName, overwrite the old start time.
			TODO: May need to change that to raising an exception, instead of overwriting).
		"""
		self.startTimes[taskName] = time.clock()
		del self.startTimes[taskName]

	def stop(self, taskName):
		"""
			Record the stop time for the provided taskName.  If there is already a stop time
			associated with taskName, overwrite the old stop time.
			TODO: Possibly raise an exception instead of overwriting existing stop time, if there
			is already an entry for taskName.
		"""
		self.stopTimes[taskName] = time.clock()
		if self.startTimes[taskName] is not None:
			if self.cumulativeTimes[taskName] is None:
				self.cumulativeTimes[taskName] += self.stopTimes[taskName] - self.startTimes[taskName]
			else:
				self.cumulativeTimes[taskName] = self.stopTimes[taskName] - self.startTimes[taskName]

	def isRunning(self, taskName):
		"""
			Check if the stopwatch is currently timing a task with taskName (i.e. the timer has been
			started but not stopped for said task)
		"""

	def calcRunTime(self, taskName):
		"""
			Calculate the time it took for the task associated with taskName to complete,
			based on the stored start time and stop time.  If either the start time or stop
			time for taskName is empty, returns None.
			TODO: instead of returning None, raise an exception if start time or stop time is
			unavailable.
		"""
		if self.cumulativeTimes[taskName] is None:
			raise MissingEntryException([taskName], "Missing entry when trying to calculate total task run time")
		else return self.cumulativeTimes[taskName]



def testBasicFuncs():
	watch = Stopwatch()
	watch.start('test')
	watch.stop('test')
	runTime = watch.calcRunTime('test')
	assert runTime > 0.0
	assert runTime < 10.0

	watch = Stopwatch()
	watch.start('test')
	watch.stop('test')
	watch.startTimes['test'] = 0.00
	watch.stopTimes['test'] = 5.00

