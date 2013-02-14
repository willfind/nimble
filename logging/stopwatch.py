"""
	The Stopwatch class is used for timing various tasks (training classifiers,
	testing classifiers)within the UML code.  Can time any generic task based
	on task name.
"""

import time


class Stopwatch(object):

	def __init__(self):
		startTimes = {}
		stopTimes = {}

	def start(self, taskName):
	"""
		Record the start time for the provided taskName.  If there is already
		a start time associated with taskName, overwrite the old start time.
		TODO: May need to change that to raising an exception, instead of overwriting).
	"""
		self.startTimes[taskName] = time.clock()

	def stop(self, taskName):
	"""
		Record the stop time for the provided taskName.  If there is already a stop time
		associated with taskName, overwrite the old stop time.
		TODO: Possibly raise an exception instead of overwriting existing stop time, if there
		is already an entry for taskName.
	"""
		self.stopTimes[taskName] = time.clock()

	def calcRunTime(self, taskName):
	"""
		Calculate the time it took for the task associated with taskName to complete,
		based on the stored start time and stop time.  If either the start time or stop
		time for taskName is empty, returns None.
		TODO: instead of returning None, raise an exception if start time or stop time is
		unavailable.
	"""
		if self.startTimes[taskName] is None or self.stopTimes[taskName] is None:
			return None
		else:
			return self.stopTimes[taskName] - self.startTimes[taskName]
