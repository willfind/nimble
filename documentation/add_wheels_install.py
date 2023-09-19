"""
Script to append the binary wheel download links table to install.rst,
using the names of the available wheels currently in the
documentation/source/wheels.

This allows us to use this script as part of the make process, instead
of as an external intervention taken at the Github Action level prior
to site building.

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
    Format a portion of the flat-table list which includes all rows with the
    same spanned OS name in the first column.
    """
    # This is counting the number of rows beyond where the definition
    # is that will be spanned over.
    spanNum = len(wheelNames) - 1
    first = True
    # the first row specified by the list must define
    # the span to combine to one cell in the OS column,
    # but only if it is actually spanning multiple lines
    ret = "  * - "
    if spanNum > 0:
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
        # See sorting comment in __main__ block for string format
        minor = nameSplit[2][3:]
        ret += f"3.{minor}\n"

        # specifying the download link with correct name
        ret += dashOnlyBuffer
        ret += f":download:`{name} <wheels/{name}>`\n"

    return ret


if __name__ == "__main__":
    # from the documentation directory (cwd enforced by make),
    # grab all wheels in the wheels folder, grab just the filename
    # instead of the full path, and convert the paths to strings
    wheelSource = pathlib.Path("source", "wheels")
    targets = wheelSource.glob('*.whl')
    targets = map(lambda x: x.parts[-1], targets)
    targets = map(os.fspath, targets)

    # wheel names in the form: "nimble-version-pyTag-pyABITag-os_arch.whl"
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

    # for possible debugging purposes when executed in Actions
    print(list(targets))
    print(byOS)

    # table generation
    toWrite = ""
    toWrite += tableStart

    # the os name we want to display generally doesn't match the OS/image
    # name given within the file name.
    #
    # dict.get used to accomodate cases where we have more or less
    # key values than expected; but as of this comment, the four
    # given should cover all built wheels.
    displayName = {'manylinux':"Linux", "manylinux2014":"Linux",
                   "linux":"Linux", "macosx":"MacOS", "win":"Windows"}
    for key, val in byOS.items():
        toWrite += rowsPerOS(displayName.get(key, key), val)

    # for possible debugging purposes when executed in Actions
    print(toWrite)

    # meant to be called by the make file, which enforces a cwd of
    # nimble/documentation
    installFile = os.path.join("source", "install.rst")
    with open(installFile, mode='a', encoding="utf-8") as f:
        f.write(toWrite)
