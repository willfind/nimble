
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

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
from .session_logger import LogID

active = None
