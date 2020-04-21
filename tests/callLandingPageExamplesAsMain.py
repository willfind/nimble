"""
Defines a single test to check the functionality of all of the
scripts contained in the examples folder.
"""

import os
import subprocess
import tempfile
import re

from nose.plugins.attrib import attr

def wifiExampleOutputMatch(output, expected):

    def decoder(line):
        return line.decode('utf-8')

    with open(expected, 'rb') as exp:
        outLines = map(decoder, output.split(b'\n'))
        expLines = map(decoder, exp.read().split(b'\n'))
        for i, (line, expLine) in enumerate(zip(outLines, expLines)):
            if i in [0, 40, 41, 42, 43]:
                continue # output on these lines will be random
            pat1 = re.compile(r'Completed\sin\s\d+\.\d{3}\sseconds.*')
            pat2 = re.compile(r'(.*)\d{4}-\d\d-\d\d\s\d\d:\d\d:\d\d')
            if re.match(pat1, line): # completion times will vary
                if not re.match(pat1, expLine):
                    return False
            elif re.search(pat2, line): # dates will vary
                preDateOut = re.search(pat2, line)
                preDateExp = re.search(pat2, expLine)
                if not preDateExp:
                    return False
                if preDateOut.group(1) != preDateExp.group(1):
                    return False
            elif line != expLine:
                return False
    return True

@attr('slow')
def test_callExamplesAsMain():
    # collect the filenames of the scripts we want to run
    examplesDir = os.path.join(os.getcwd(), 'documentation', 'landingPage')
    examplesFiles = [f for f in os.listdir(examplesDir) if f.endswith('.py')]

    results = {}
    scriptsWithPlots = ['shoppers_train.py', 'shoppers_explore.py']
    for script in examplesFiles:
        scriptLoc = os.path.join(examplesDir, script)
        # Copy the script, commenting out plotting functions
        if script in scriptsWithPlots:
            tempFile = tempfile.NamedTemporaryFile('w+')
            with open(scriptLoc) as f:
                plotStart = re.compile(r'\.plot.*\(')
                openParen = 0
                for line in f.readlines():
                    if openParen or re.search(plotStart, line):
                        tempFile.write('# ' + line)
                        openParen += line.count('(')
                        openParen -= line.count(')')
                    else:
                        tempFile.write(line)

            scriptLoc = tempFile.name
            tempFile.seek(0)

        cmd = ("python", scriptLoc)
        out = subprocess.PIPE
        err = subprocess.PIPE

        # We want these scripts to run with the local copy of nimble, so we need
        # the current working directory (as established by runTests) to be
        # on the path variable in the subprocess. However, we also want the
        # environment to otherwise be the same (because we know it works).
        # Therefore we reuse the environment, except with a modification to
        # PYTHONPATH
        env = os.environ
        env['PYTHONPATH'] = os.getcwd()
        cp = subprocess.run(cmd, stdout=out, stderr=err, cwd=examplesDir, env=env)
        results[script] = cp
        if script in scriptsWithPlots:
            tempFile.close()

    print("")
    print("*** Results ***")
    print("")
    print("")
    fail = False
    sortedKeys = sorted(results.keys())
    for key in sortedKeys:
        cp = results[key]
        outputFile = key[:-3] + '_output.txt'
        expOut = os.path.join(examplesDir, outputFile)
        if cp.returncode != 0:
            fail = True
            print(key + " : " + str(cp.stderr))
            print("")
        elif key == 'wifi_support.py':
            # wifi support uses randomness and logging the output will vary
            # in some places. so we need to check output line by line.
            fail = not wifiExampleOutputMatch(cp.stdout, expOut)
            if fail:
                msg = key + " : Did not match output in "
                msg += outputFile
                print(msg)
        else:
            with open(expOut, 'rb') as exp:
                expRead = exp.read()
                if cp.stdout != expRead:
                    print(cp.stdout)
                    print(expRead)
                    fail = True
                    print(key + " : Did not match output in " + outputFile)
    assert not fail
