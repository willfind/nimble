"""
Short module demonstrating the flatter imports and user facing functions provided
in the root of the UML package

"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	from UML import run
	from UML import normalizeData
	from UML import createData

	variables = ["x1","x2","x3"]
	data1 = [[1,0,0], [3,3,3], [5,0,0],]
	trainObj = createData('Matrix', data1, variables)

	data2 = [[1,0,0],[1,1,1],[5,1,1], [3,4,4]]
	testObj = createData('Matrix', data2, variables)

	normalizeData('mlpy.PCA', trainObj, testObj, arguments={'k':2})
	ret = run('sciKitLearn.KMeans', trainObj, testObj, arguments={'n_clusters':3})

	print "returned: " + str(ret)
	# clustering returns a row vector of indices, referring to the cluster centers,
	# we don't care about the exact numbers, this verifies that the appropriate
	# ones are assigned to the same clusters
	assert ret.data[0,0] == ret.data[1,0]
	assert ret.data[0,0] != ret.data[2,0]
	assert ret.data[0,0] != ret.data[3,0]
	assert ret.data[2,0] != ret.data[3,0]
