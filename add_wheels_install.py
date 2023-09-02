"""
Script to append the binary wheel download links table to install.rst,
using the names of the available wheels in the current directory.

For use in github actions

"""

import pathlib
import os

# We define this up here that it's easier to see the required indenting
# within the string literal.
tableStart = """
.. flat-table::
  :align: left
  :widths: auto
  :header-rows: 1
  
  * - OS
    - Python Version
    - Architecture

"""

def rowsPerOS(osName, wheelNames):
    """
    Format a portion of the list table which includes all rows with the same
    spanned OS name in the first column
    """
    # This is counting the number of rows beyond where the definition
    # is that will be spanned over.
    spanNum = len(wheelNames) - 1
    first = True
    # the first row specified by the list must define
    # the span to combine to one cell in the OS column,
    # but only if it is actually spanning multiple lines
    ret = "  * - "
    if spanNum > 1:
        ret += f":rspan:`{spanNum}` "
    ret += f"{osName}\n"

    for name in wheelNames:
        nameSplit = name.split('-')
        dashOnlyBuffer = "    - "
        # See header row defined in the tableStart for standard format
        if first:
            ret += dashOnlyBuffer
            first = False
        else:
            ret += "  * - "

        # grab the minor part of the python version from "cp3*""
        # See sorting comment for name format
        minor = nameSplit[2][3:]
        ret += f"3.{minor}\n"

        # specifying the download link with correct name
        ret += dashOnlyBuffer
        ret += f":download:`{name} <wheels/{name}>`\n"

    return ret


if __name__ == "__main__":
    # in the current directory, grab all wheels, put them into a list,
    # and convert the paths to strings
    cwd = pathlib.Path(".")
    targets = list(cwd.glob('*.whl'))
    targets = map(os.fspath, targets)

    # wheel names are of the form: "nimble-version-pyVersion-pyVersion-os_arch.whl"
    # So first we sort by the python version, then they are partitioned into
    # lists per each OS
    verSorted = sorted(targets, key=lambda x: int(x.split('-')[2][2:]))
    byOS = {}
    for wheelName in verSorted:
        parts = wheelName.split("-")
        currOS = parts[-1].split("_")[0]
        if currOS not in byOS:
            byOS[currOS] = [wheelName]
        else:
            byOS[currOS].append(wheelName)

    # table generation
    toWrite = ""
    toWrite += tableStart

    # the os display name generally doesn't match the OS/image name given
    # within the file name
    toWrite += rowsPerOS("Linux", byOS['linux'])
    toWrite += rowsPerOS("MacOS", byOS['macosx'])
    toWrite += rowsPerOS("Windows", byOS['win'])

    print(toWrite)

    installFile ="./documentation/source/install.rst"
    with open(installFile, mode='a', encoding="utf-8") as f:
        f.write(toWrite)
