"""
Script to remove all .py and .c source files bundled in a wheel

Takes two args:
1) the directory to place the new wheel
2) the full path (directory + name) to the wheel

Example: when in the same directory

python remove_sourece.py . ./foo.whl

If the wheel is within the target directory, it will be written over.
"""

import subprocess
import os
import sys


def recursiveRemove(root):
    for root, dirs, names in os.walk(root):
        for subDir in dirs:
            recursiveRemove(os.path.join(root, subDir))
        for name in names:
            if name.endswith(".py") or name.endswith('.c'):
                fullPath = os.path.join(root, name)
                os.remove(fullPath)

if __name__ == "__main__":
    targetDir = sys.argv[1]
    target = sys.argv[2]
    version = target.split('-')[1]

    cmd = ["wheel", "unpack", target, "-d", targetDir]
    compProc = subprocess.run(cmd, check=True)

    pkName = "nimble-" + version
    pkPath = os.path.join(targetDir, pkName)
    recursiveRemove(pkPath)
    cmd = ["wheel", "pack", pkPath, "-d", targetDir]
    compProc = subprocess.run(cmd, check=True)
