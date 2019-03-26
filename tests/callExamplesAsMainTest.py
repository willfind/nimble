"""
Defines a single test to check the functionality of all of the
scripts contained in the examples folder.

"""

import os
import subprocess
import tempfile

def test_callAllAsMain():
    """
    Calls each script in examples, confirms it completes without an exception.

    """
    # collect the filenames of the scripts we want to run
    examplesDir = os.path.join(os.getcwd(), 'examples')
    examplesFiles = os.listdir(examplesDir)
    cleaned = []
    for fileName in examplesFiles:
        if fileName.endswith('.py'):
            cleaned.append(fileName)

    results = {}
    for script in cleaned:
        scriptLoc = os.path.join(examplesDir, script)
        # Provide a dummy output directory argument. For the plotting example,
        # this will write the files into the temp dir instead of generating
        # plots on the screen.
        tempOutDir = tempfile.mkdtemp()

        cmd = ("python", scriptLoc, tempOutDir)
        spP = subprocess.PIPE

        # We want these scripts to run with the local copy of UML, so we need
        # the current working directory (as established by runTests) to be
        # on the path variable in the subprocess. However, we also want the
        # environment to otherwise be the same (because we know it works).
        # Therefore we reuse the environment, except with a modification to
        # PYTHONPATH
        env = os.environ
        env['PYTHONPATH'] = os.getcwd()
        cp = subprocess.run(cmd, stdout=spP, stderr=spP, cwd=os.getcwd(), env=env)
        results[script] = cp

    print("")
    print("*** Results ***")
    print("")
    print("")
    fail = False
    sortedKeys = sorted(results.keys())
    for key in sortedKeys:
        cp = results[key]
        if cp.returncode != 0:
            fail = True
        print(key + " : " + str(cp))
        print("")
    assert not fail
