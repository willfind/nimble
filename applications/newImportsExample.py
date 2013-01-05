"""
Short module demonstrating the flatter imports and user facing functions provided
in the root of the UML package

"""


# PEP 366 'boilerplate', plus the necessary import of the top level package
if __name__ == "__main__" and __package__ is None:
	import sys
	# add UML parent directory to sys.path
	sys.path.append(sys.path[0].rsplit('/',2)[0])
	import UML
	import UML.applications
	__package__ = "UML.applications"

if __name__ == "__main__":

	from UML import run
	from UML import data

	variables = ["x1","x2"]
	data1 = [[1,0], [3,3], [5,0],]
	trainingObj = data('DenseMatrixData', data1, variables)

	data2 = [[1,0],[1,1],[5,1], [3,4]]
	testObj = data('DenseMatrixData', data2)

	ret = run('sciKitLearn', "KMeans", trainingObj, testObj, output=None, arguments={'n_clusters':3})

	# clustering returns a row vector of indices, referring to the cluster centers,
	# we don't care about the exact numbers, this verifies that the appropriate
	# ones are assigned to the same clusters
	assert ret.data[0,0] == ret.data[1,0]
	assert ret.data[0,0] != ret.data[2,0]
	assert ret.data[0,0] != ret.data[3,0]
	assert ret.data[2,0] != ret.data[3,0]
