"""
The Stopwatch class is used for timing various tasks (training
classifiers, testing classifiers) within the nimble code.  Can time any
generic task based on task name.
"""

import time

class Stopwatch(object):
    """
    The Stopwatch class is used for timing various tasks (training
    classifiers, testing classifiers) within the nimble code.  Can time
    any generic task based on task name.
    """
    def __init__(self):
        self.startTimes = dict()
        self.stopTimes = dict()
        self.cumulativeTimes = dict()
        self.isRunningStatus = dict()

    def start(self, taskName):
        """
        Record the start time for the provided taskName.  If there is
        already a start time associated with taskName, overwrite the old
        start time.
        TODO: May need to change that to raising an exception,
        instead of overwriting).
        """
        if (taskName in self.isRunningStatus
                and self.isRunningStatus[taskName]):
            msg = "Task: " + taskName + " has already been started."
            raise TypeError(msg)
        else:
            self.startTimes[taskName] = time.process_time()
            if taskName not in self.cumulativeTimes:
                self.cumulativeTimes[taskName] = 0.0
            self.isRunningStatus[taskName] = True

    def stop(self, taskName):
        """
        Record the stop time for the provided taskName.  If there is
        already a stop time associated with taskName, overwrite the
        old stop time.
        TODO: Possibly raise an exception instead of overwriting
        existing stop time, if there is already an entry for taskName.
        """
        if (taskName not in self.startTimes
                or taskName not in self.isRunningStatus):
            msg = "Tried to stop task '" + taskName
            msg += "' that was not started in Stopwatch.stop()"
            raise TypeError(msg)
        elif not self.isRunningStatus[taskName]:
            raise TypeError("Unable to stop task that has already stopped")

        self.stopTimes[taskName] = time.process_time()
        self.isRunningStatus[taskName] = False
        if taskName in self.cumulativeTimes:
            self.cumulativeTimes[taskName] += (self.stopTimes[taskName]
                                               - self.startTimes[taskName])
        else:
            self.cumulativeTimes[taskName] = (self.stopTimes[taskName]
                                              - self.startTimes[taskName])

    def reset(self, taskName, resetCumulativeTime=True):
        """
        Reset/remove all values in stopwatch associated with taskName.
        If resetCumulativeTime is True, get rid of the running total of
        time for taskName.  If resetCumulativeTime is false, the running
        total is preserved but taskName will be stopped and
        isRunning(taskName) will return False.
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
        Check if the stopwatch is currently timing a task with taskName
        (i.e. the timer has been started but not stopped for said task).
        Returns true if start() has been called for this taskName but
        neither stop() nor reset() have been called for this taskName.
        Otherwise returns false.
        """
        if taskName in self.isRunningStatus:
            return self.isRunningStatus[taskName]
        return False

    def calcRunTime(self, taskName):
        """
        Calculate the time it took for the task associated with taskName
        to complete, based on the stored start time and stop time.  If
        the timer is still timing the task associated with taskName
        (i.e. isRunning(taskName) is True), raises an
        ImproperActionException.
        """
        if (taskName not in self.cumulativeTimes
                or taskName not in self.isRunningStatus):
            msg = "Missing entry when trying to calculate total task "
            msg += "run time: " + str(taskName)
            raise TypeError(msg)
        elif self.isRunningStatus[taskName]:
            msg = 'Can\'t calculate total running time for ' + taskName
            msg += ', as it is still running'
            raise TypeError(msg)
        else:
            return self.cumulativeTimes[taskName]
