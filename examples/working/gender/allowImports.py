"""
Defines function to perform the necessary setup for making a script
called from this folder aware of the package structure around it

"""


# PEP 366 'boilerplate', plus the necessary import of the top level package
def boilerplate():
    import sys
    import os
    # add UML parent directory to sys.path
    thisFileLocation = os.path.abspath(__file__)
    UMLParentDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(thisFileLocation))))
    sys.path.append(UMLParentDir)
    import UML
    import UML.examples
    import UML.examples.working
    import UML.examples.working.gender
    __package__ = "UML.examples.working.gender"
