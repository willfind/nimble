"""
Defines a single test to check the functionality of all of the
scripts contained in the examples folder.



"""

import os
import sys
import StringIO
import tempfile
import shutil
import copy

import UML
#ensures UML.examples.allowImports is in sys.modules
import UML.examples.allowImports


def test_callAllAsMain():
    """
    Calls each script in examples, confirms it completes without an exception.

    """

    # Bind the name allowImports to the appropriate, already loaded, module.
    # Needed because each script we call imports a function from allowImports,
    # but since we are calling from a different context, the import as given
    # fails.
    sys.modules['allowImports'] = sys.modules['UML.examples.allowImports']

    # collect the filenames of the scripts we want to run
    examplesDir = os.path.join(UML.UMLPath, 'examples')
    examplesFiles = os.listdir(examplesDir)
    cleaned = []
    for fileName in examplesFiles:
        if fileName.endswith('.py') and not fileName.startswith("__"):
            cleaned.append(fileName)

    # so when we execute the scripts, we actually run the poriton that is meant
    # to be run when called as main
    __name__ = "__main__"

    # we want to capture stout and sterr; this can be adjusted for debugging,
    # but without this it's challenging to check the results
    results = {}
    examplesSTOUT = StringIO.StringIO()
    examplesSTERR = StringIO.StringIO()
    tempOutDir = tempfile.mkdtemp()
    backupSTOUT = sys.stdout
    backupSTERR = sys.stderr
    backuputARGV = sys.argv
    try:
        sys.stdout = examplesSTOUT
        sys.stderr = examplesSTERR
        sys.argv = copy.copy(sys.argv)
        sys.argv[1] = tempOutDir

        for script in cleaned:
            try:
                execfile(os.path.join(examplesDir, script))
                results[script] = "Success"
            except Exception:
                results[script] = sys.exc_info()
    finally:
        sys.stdout = backupSTOUT
        sys.stderr = backupSTERR
        sys.argv = backuputARGV
        shutil.rmtree(tempOutDir)

    print ""
    print "*** Results ***"
    print ""
    print ""
    fail = False
    sortedKeys = sorted(results.keys())
    for key in sortedKeys:
        val = results[key]
        if val != "Success":
            fail = True
        print key + " : " + str(val)
        print ""
    assert not fail
    #if isinstance(val, tuple) and len(val) > 0 and isinstance(val[0], Exception):
    #raise val[1][1], None, val[1][2]
    #print key
    #print val[1][1], None, val[1][2]
