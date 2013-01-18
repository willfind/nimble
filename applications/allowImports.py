"""
Defines function to perform the necessary setup for making a script
called from this folder aware of the package structure around it

"""


# PEP 366 'boilerplate', plus the necessary import of the top level package
def boilerplate():
	import sys
	# add UML parent directory to sys.path
	sys.path.append(sys.path[0].rsplit('/',2)[0])
	import UML
	import UML.applications
	__package__ = "UML.applications"

