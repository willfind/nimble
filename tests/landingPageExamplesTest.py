"""
Defines a single test to check the functionality of all of the
scripts contained in the examples folder.
"""

import os
import subprocess
import tempfile
import re

import pytest

@pytest.mark.slow
def test_callExamplesAsMain():
    # collect the filenames of the scripts we want to run
    examplesDir = os.path.join(os.getcwd(), 'documentation', 'source',
                               'examples')
    examplesFiles = [f for f in os.listdir(examplesDir) if f.endswith('.py')]
    results = {}
    scriptsWithPlots = ['unsupervised_learning.py', 'exploring_data.py']

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
    for key in results.keys():
        cp = results[key]
        outputFile = key[:-3] + '_output.txt'
        expOut = os.path.join(examplesDir, 'outputs', outputFile)
        if cp.returncode != 0:
            failures.append(key)
            print(key + " : ", cp.stderr.decode('utf-8'))
            print("")
        else:
            outLines = cp.stdout.split(b'\n')
            if key == 'neural_networks.py':
                # check only final line output, ignore intermediate updates
                outLines = [l.split(b'\r')[-1] for l in outLines]
            with open(expOut, 'rb') as exp:
                expLines = exp.readlines()
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
                        failures.append(key)
                        print(key + " : Did not match output in " + outputFile)
                        print('  Discrepancy found in line {i}'.format(i=i))
                        break

    assert not failures
