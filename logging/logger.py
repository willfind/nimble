import numpy

"""
	Handle logging of creating and testing classifiers.  Currently
	creates two versions of a log for each run:  one that is human-readable,
	and one that is machine-readable (csv).  
	Should report, for each run:
		Size of input data
			# of features  (columns)
			# of points (rows)
			# points used for training
			# points used for testing

		Name of package (mlpy, scikit learn, etc.)
		Name of algorithm
		