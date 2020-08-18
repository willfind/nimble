"""
Defines a single test to check the functionality of all of the
scripts contained in the examples folder.
"""

import os
import subprocess
import tempfile
import re

from nose.plugins.attrib import attr


def getScriptOutputChecker(script):
    if script == 'digits_train.py':
        return digitsExampleOutputMatch
    raise ValueError('no script checker for ' + script)

def decoder(line):
    return line.decode('utf-8')

def digitsExampleOutputMatch(output, expected):
    """
    Validate format of Keras output ignoring variable sections and
    accounting for continuous updates.
    """
    outLines = list(map(decoder, re.split(b'\n\r\b', output)))
    # There is no expected output file because Keras does not allow for
    # randomness control and updates continuously. It is even unclear how many
    # lines the output will contain. Instead we will verify that the format
    # matches our expectations
    newEpoch = re.compile('Epoch [0-9]{1,2}/10')
    epochStatus = re.compile(r'[0-9]{1,4}/1195 \[[=>\.]{30}\]')
    expectEpochStatus = False
    for i, line in enumerate(outLines):
        line = line.strip()
        # most lines will be epoch status updates
        if not re.match(epochStatus, line):
            # Non-blank lines must indicate new epoch, or be a (float) result
            newEpochLine = re.match(newEpoch, line)
            if line and not newEpochLine:
                try:
                    resultLine = 0 <= float(line) <= 1
                except ValueError:
                    return False

    return True

@attr('slow')
def test_callExamplesAsMain():
    # collect the filenames of the scripts we want to run
    examplesDir = os.path.join(os.getcwd(), 'documentation', 'source',
                               'examples')
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

        # We want these scripts to run with the local copy of nimble, so we
        # need the current working directory (as established by runTests) to be
        # on the path variable in the subprocess. However, we also want the
        # environment to otherwise be the same (because we know it works).
        # Therefore we reuse the environment, except with a modification to
        # PYTHONPATH
        env = os.environ
        env['PYTHONPATH'] = os.getcwd()
        cp = subprocess.run(cmd, stdout=out, stderr=err, cwd=examplesDir,
                            env=env)
        results[script] = cp
        if script in scriptsWithPlots:
            tempFile.close()

    print("")
    print("*** Results ***")
    print("")

    failures = []
    variableOutputScripts = ['digits_train.py']
    for key in results.keys():
        cp = results[key]
        outputFile = key[:-3] + '_output.txt'
        expOut = os.path.join(examplesDir, 'outputs', outputFile)
        if cp.returncode != 0:
            failures.append(key)
            print(key + " : " + str(cp.stderr))
            print("")
        elif key not in variableOutputScripts:
            with open(expOut, 'rb') as exp:
                expLines = exp.readlines()
                outLines = cp.stdout.split(b'\n')
                for i, (out, exp) in enumerate(zip(outLines, expLines)):
                    # remove trailing whitespace
                    out = out.rstrip()
                    exp = exp.rstrip()
                    if exp.startswith(b'REGEX: '):
                        exp = exp[7:]
                        match = re.match(exp, out)
                    else:
                        match = exp == out
                    if not match:
                        print(out, exp)
                        failures.append(key)
                        print(key + " : Did not match output in " + outputFile)
                        print('  Discrepancy found in line {i}'.format(i=i))
                        break

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
