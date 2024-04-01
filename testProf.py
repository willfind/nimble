
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

import os
import pstats
import time
import cProfile
from six.moves import range

gFileName = None

def report(fileName=None, sort=None, n=25, stripDirs=False):
    """
    This function is to read a report and sort it by either cumtime and tottime
    and then display top rows.

    If fileName is None, then it will read gFileName.
    """
    global gFileName
    if fileName is None:
        fileName = gFileName
    elif gFileName is None:
        gFileName = fileName
    p = pstats.Stats(fileName)
    if sort is not None:
        if stripDirs:
            p.strip_dirs().sort_stats(sort).print_stats(n)
        else:
            p.sort_stats(sort).print_stats(n)
    else:
        if stripDirs:
            p.strip_dirs().sort_stats('cumtime').print_stats(n)
            p.strip_dirs().sort_stats('tottime').print_stats(n)
        else:
            p.sort_stats('cumtime').print_stats(n)
            p.sort_stats('tottime').print_stats(n)

def runTests(fileName=None):
    """
    This function is to run
    python runTests.py
    and store cprofile report in fileName
    """
    global gFileName
    if fileName is None:
        ct = time.localtime()
        fileName = 'runTestsProf%s%s%s%s%s%s' % (ct.tm_year, ct.tm_mon, ct.tm_mday, ct.tm_hour, ct.tm_min, ct. tm_sec)
    s = 'python -m cProfile -o %s runTests.py'%fileName
    gFileName = fileName
    os.system(s)

def dataTests(fileName=None, test=''):
    """
    This function is to run
    nosetests testObjects.py
    or
    nosetests testObjects.py:xxxx

    xxxx is the string in test

    fileName is the name of the file that the cprofile report will be stored
    """
    global gFileName
    if fileName is None:
        ct = time.localtime()
        fileName = 'dataTestsProf%s%s%s%s%s%s' % (ct.tm_year, ct.tm_mon, ct.tm_mday, ct.tm_hour, ct.tm_min, ct. tm_sec)
    if test:
        test = ':'+test
    s = 'nosetests --with-cprofile --cprofile-stats-file=%s data/tests/testObjects.py%s' % (fileName, test)
    gFileName = fileName
    os.system(s)

#other ways:
# import cProfile
# x = range(1000)
# pr = cProfile.Profile()
# pr.enable()
#
# for i in x:
# 	call_function()
#
# pr.disable()
# pr.print_stats(sort='time')

def funcTests(f, n=1000):
    x = list(range(n))
    pr = cProfile.Profile()
    pr.enable()

    for i in x:
        f()

    pr.disable()
    pr.print_stats(sort='time')