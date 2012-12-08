"""
Module defining custom exceptions to be used in this package.
"""

class ArgumentException (Exception):
	"""
	Exception to be thrown when the value of an argument of some function
	renders it impossible to complete the operation that function is meant
	to perform.
	"""

	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)


