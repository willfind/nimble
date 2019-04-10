"""
Attributes callable directly from UML.logger.

active logger is set upon UML instantiation.
"""

from __future__ import absolute_import
from .data_set_analyzer import produceFeaturewiseReport
from .data_set_analyzer import produceAggregateReport
from .uml_logger import UmlLogger, initLoggerAndLogConfig
from .uml_logger import handleLogging
from .uml_logger import stringToDatetime
from .stopwatch import Stopwatch

active = None

__all__ = ['active']
