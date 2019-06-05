"""
Attributes callable directly from nimble.logger.

active logger is set upon nimble instantiation.
"""

from __future__ import absolute_import
from .data_set_analyzer import produceFeaturewiseReport
from .data_set_analyzer import produceAggregateReport
from .session_logger import SessionLogger, initLoggerAndLogConfig
from .session_logger import handleLogging
from .session_logger import startTimer, stopTimer
from .session_logger import stringToDatetime
from .stopwatch import Stopwatch

active = None

__all__ = ['active']
