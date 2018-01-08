"""
Defines a single test to check the functionality of all of the
scripts contained in the examples folder.



"""

from __future__ import absolute_import
from __future__ import print_function
import os
import sys
try:
    from StringIO import StringIO#python 2
except:
    from six import StringIO#python 3
import tempfile
import shutil
import copy

import UML
#ensures UML.examples.allowImports is in sys.modules
import UML.examples.allowImports
import warnings


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
    examplesSTOUT = StringIO()
    examplesSTERR = StringIO()
    tempOutDir = tempfile.mkdtemp()
    backupSTOUT = sys.stdout
    backupSTERR = sys.stderr
    backuputARGV = sys.argv
    try:
        sys.stdout = examplesSTOUT
        sys.stderr = examplesSTERR
        sys.argv = copy.copy(sys.argv)
        if len(sys.argv) > 1:
            sys.argv[1] = tempOutDir
        else:
            sys.argv.append(tempOutDir)

        # Since they sometimes have file IO side effects to files in the
        # repo, and we don't want those effects to be reflected in changes
        # to the repo, we fix a random seed for all future calls.
        # Since The example scripts are demonstartions of api usage NOT
        # randomized unit tests, this has no effect on the usefulness of
        # this test.
        UML.randomness.startAlternateControl(1)

        for script in cleaned:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if sys.version_info.major <= 2:
                        execfile(os.path.join(examplesDir, script))
                    else:
                        exec(compile(open(os.path.join(examplesDir, script)).read(), os.path.join(examplesDir, script), 'exec'))
                results[script] = "Success"
            except Exception:
                results[script] = sys.exc_info()
    except Exception:
        pass
    finally:
        sys.stdout = backupSTOUT
        sys.stderr = backupSTERR
        sys.argv = backuputARGV
        shutil.rmtree(tempOutDir)
        UML.randomness.endAlternateControl()

    print("")
    print("*** Results ***")
    print("")
    print("")
    fail = False
    sortedKeys = sorted(results.keys())
    for key in sortedKeys:
        val = results[key]
        if val != "Success":
            fail = True
        print(key + " : " + str(val))
        print("")
    assert not fail
    #if isinstance(val, tuple) and len(val) > 0 and isinstance(val[0], Exception):
    #raise val[1][1], None, val[1][2]
    #print key
    #print val[1][1], None, val[1][2]
