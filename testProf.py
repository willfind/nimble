import os
import pstats
import time

gFileName = None

def report(fileName=None, sort=None, n=25, stripDirs=True):
    """

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