"""
Defines a single test to check the functionality of all of the
scripts contained in the examples folder.
"""

import os
import subprocess
import tempfile
import re

from nose.plugins.attrib import attr

@attr('slow')
def test_callExamplesAsMain():
    # collect the filenames of the scripts we want to run
    examplesDir = os.path.join(os.getcwd(), 'documentation', 'landingPage')
    examplesFiles = [f for f in os.listdir(examplesDir) if f.endswith('.py')]

    results = {}
    scriptsWithPlots = ['shoppers_train.py', 'shoppers_explore.py']

    for script in ['digits_train.py']:
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
        cp.stdout.flush()
        print(cp.stdout)
        assert False
        results[script] = cp
        if script in scriptsWithPlots:
            tempFile.close()

    print("")
    print("*** Results ***")
    print("")

    failures = []
    variableOutputScripts = ['digits_train.py', 'wifi_support.py']
    for key in results.keys():
        cp = results[key]
        outputFile = key[:-3] + '_output.txt'
        expOut = os.path.join(examplesDir, outputFile)
        if cp.returncode != 0:
            failures.append(key)
            print(key + " : " + str(cp.stderr))
            print("")
        elif key not in variableOutputScripts:
            with open(expOut, 'rb') as exp:
                expRead = exp.read()
                if cp.stdout != expRead:
                    failures.append(key)
                    print(cp.stdout)
                    print(expRead)
                    print(key + " : Did not match output in " + outputFile)
        # The following scripts have variable outputs so each requires its
        # own function for validating the output
        else:
            outputChecker = getScriptOutputChecker(key)
            fail = not outputChecker(cp.stdout, expOut)
            if fail:
                failures.append(key)
                msg = key + " : Did not match output in "
                msg += outputFile
                print(msg)

    assert not failures

def getScriptOutputChecker(script):
    if script == 'wifi_support.py':
        return wifiExampleOutputMatch
    if script == 'digits_train.py':
        return digitsExampleOutputMatch
    raise ValueError('no script checker for ' + script)

def decoder(line):
    return line.decode('utf-8')

def wifiExampleOutputMatch(output, expected):
    """
    Ignore completely random lines and validate logging format but
    ignore specific timestamps.
    """
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

def digitsExampleOutputMatch(output, expected):
    """
    Validate format of Keras output ignoring variable sections and
    accounting for continuous updates.
    """
    outLines = map(decoder, output.split(b'\n'))
    # There is no expected output file because Keras does not allow for
    # randomness control and updates continuously. It is even unclear how many
    # lines the output will contain. Instead we will verify that the format
    # matches our expectations
    newEpoch = re.compile(b'Epoch [0-9]{1,2}/10')
    epochStatus = re.compile(r'[0-9]{1,4}/1195 \[[=\.]{30}\]')
    expectEpochStatus = False
    for i, line in enumerate(outLines):
        # most lines will be epoch status updates
        if not re.match(epochStatus, line):
            # Non-blank lines must indicate new epoch, or be a (float) result
            newEpochLine = re.match(newEpoch, line)
            resultLine = 0 <= float(line) <= 1
            if line and not (newEpochLine or resultLine):
                return False

    return True
