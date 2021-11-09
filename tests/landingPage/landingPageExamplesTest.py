"""
Defines a single test to check the functionality of all of the
scripts contained in the examples folder.
"""

import os
import subprocess
import tempfile
import re
import shutil

import pytest

EXDIR = os.path.join(os.getcwd(), 'documentation', 'source', 'examples')

def back_singleExample(scriptLoc):
    """
    Execute the script at the given location, and return the CompletedProcess
    """
    # Copy the script, to use the same terminal size used in examples and
    # comment out plotting functions
    with open(scriptLoc) as f:
        with tempfile.NamedTemporaryFile('w+') as tempFile:
            # set a terminal size that will display the data as we would like
            terminalSize = [
                "import os",
                "import shutil",
                "size = os.terminal_size((132, 30))",
                "shutil.get_terminal_size = lambda *args, **kwargs: size"
                ]
            tempFile.write('\n'.join(terminalSize))
            tempFile.write('\n')
            tempFile.write(f.read())

            tempFile.seek(0)

            cmd = ("python", tempFile.name)
            out = subprocess.PIPE
            err = subprocess.PIPE

            # We want these scripts to run with the local copy of nimble, so we
            # need the curren(t working directory as established by runTests)
            # to be on the path variable in the subprocess. However, we also
            # want the environment to otherwise be the same (because we know it
            # works). Therefore we reuse the environment, except with a
            # modification to PYTHONPATH
            env = os.environ
            env['PYTHONPATH'] = os.getcwd()
            return subprocess.run(cmd, stdout=out, stderr=err, cwd=EXDIR,
                                  env=env)

def back_singleExample_withPlots(scriptLoc):
    """
    Modify the script to write plots out to a temp file, then excute
    """
    tmpNTF = tempfile.NamedTemporaryFile
    plotStart = re.compile(r'\.plot.*\(')

    with open(scriptLoc) as f, tmpNTF('w+') as tSrc, tmpNTF('w+') as tPlot:
        openParen = False
        seenShow = False
        for line in f.readlines():
            if openParen or re.search(plotStart, line):
                openParen = True
                if 'show' in line:
                    line.replace('show=True', 'show=False')
                    seenShow = True
                if ')' in line:
                    useComma = line[line.index(')')-1] != '('
                    changeTxt = "outPath='{}')".format(tPlot.name)
                    if not seenShow:
                        changeTxt = "show=False, " + changeTxt
                    if useComma:
                        changeTxt = ', ' + changeTxt
                    line = line.replace(')', changeTxt)
                    openParen = False
                    seenShow = False
            tSrc.write(line)

        scriptLoc = tSrc.name
        tSrc.seek(0)

        return back_singleExample(tSrc.name)

def back_callExampleAsMain(script):
    # check which backend needed
    scriptsWithPlots = ['unsupervised_learning.py', 'exploring_data.py']

    scriptLoc = os.path.join(EXDIR, script)
    if script in scriptsWithPlots:
        cp = back_singleExample_withPlots(scriptLoc)
    else:
        cp = back_singleExample(scriptLoc)

    outputFile = script[:-3] + '_output.txt'
    expOut = os.path.join(os.getcwd(), 'tests', 'landingPage', outputFile)
    assert cp.returncode == 0

    outLines = cp.stdout.split(b'\n')
    if script == 'neural_networks.py':
        # check only final line output, ignore intermediate updates
        outLines = [l.split(b'\r')[-1] for l in outLines]

    with open(expOut, 'rb') as exp:
        expLines = exp.readlines()
        for out, exp in zip(outLines, expLines):
            print(out)
            print(exp)
            # remove trailing whitespace
            out = out.rstrip()
            exp = exp.rstrip()
            if exp.startswith(b'REGEX: '):
                exp = exp[7:]
                assert re.match(exp, out)
            else:
                assert exp == out

@pytest.mark.slow
def test_examples_additional_functionality():
    script = 'additional_functionality.py'
    back_callExampleAsMain(script)

@pytest.mark.slow
def test_examples_cleaning_data():
    script = 'cleaning_data.py'
    back_callExampleAsMain(script)

@pytest.mark.slow
def test_examples_exploring_data():
    script = 'exploring_data.py'
    back_callExampleAsMain(script)

@pytest.mark.slow
def test_examples_merging_and_tidying_data():
    script = 'merging_and_tidying_data.py'
    back_callExampleAsMain(script)

@pytest.mark.slow
def test_examples_neural_networks():
    script = 'neural_networks.py'
    back_callExampleAsMain(script)

@pytest.mark.slow
def test_examples_supervised_learning():
    script = 'supervised_learning.py'
    back_callExampleAsMain(script)

@pytest.mark.slow
def test_examples_unsupervised_learning():
    script = 'unsupervised_learning.py'
    back_callExampleAsMain(script)
