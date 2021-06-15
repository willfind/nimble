"""
Attributes callable directly from nimble.core.logger.

active logger is set upon nimble instantiation.
"""

from .session_logger import SessionLogger, initLoggerAndLogConfig
from .session_logger import log, showLog
from .session_logger import handleLogging
from .session_logger import loggingEnabled
from .session_logger import deepLoggingEnabled
from .session_logger import stringToDatetime

active = None
