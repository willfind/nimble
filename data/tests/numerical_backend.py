
import numpy

from nose.tools import *

import UML

from UML.exceptions import ArgumentException
from UML.exceptions import ImproperActionException

from UML.data.tests.baseObject import DataTestObject

from UML.randomness import numpyRandom
from UML.randomness import pythonRandom

def calleeConstructor(data, constructor):
	if constructor is None:
		return pythonRandom.random()
	else:
		return constructor(data)

def back_otherObjectExceptions(callerCon, op):
	""" Test operation raises exception when param is not a UML data object """
	data = [[1,2,3], [4,5,6], [7,8,9]]
	caller = callerCon(data)
	toCall = getattr(caller, op)
	toCall({1:1, 2:2, 3:'three'})


def back_selfNotNumericException(callerCon, calleeCon, op):
	""" Test operation raises exception if self has non numeric data """
	data1 = [['why','2','3'], ['4','5','6'], ['7','8','9']]
	data2 = [[1,2,3], [4,5,6], [7,8,9]]
	try:
		caller = callerCon(data1)
		callee = calleeConstructor(data2, calleeCon)
	except:
		raise ArgumentException("Data type doesn't support non numeric data")
	toCall = getattr(caller, op)
	toCall(callee)

def back_otherNotNumericException(callerCon, calleeCon, op):
	""" Test elementwiseMultiply raises exception if param object has non numeric data """
	data1 = [[1,2,3], [4,5,6], [7,8,9]]
	data2 = [['one','2','3'], ['4','5','6'], ['7','8','9']]
	caller = callerCon(data1)
	callee = calleeConstructor(data2, UML.data.List)

	toCall = getattr(caller, op)
	toCall(callee)

def back_pShapeException(callerCon, calleeCon, op):
	""" Test operation raises exception the shapes of the object don't fit correctly """
	data1 = [[1,2,6], [4,5,3], [7,8,6]]
	data2 = [[1,2,3], [4,5,6], ]
	caller = callerCon(data1)
	callee = calleeConstructor(data2, calleeCon)

	toCall = getattr(caller, op)
	toCall(callee)

def back_fShapeException(callerCon, calleeCon, op):
	""" Test operation raises exception the shapes of the object don't fit correctly """
	data1 = [[1,2], [4,5], [7,8]]
	data2 = [[1,2,3], [4,5,6], [7,8,9]]
	caller = callerCon(data1)
	callee = calleeConstructor(data2, calleeCon)

	toCall = getattr(caller, op)
	toCall(callee)

def back_pEmptyException(callerCon, calleeCon, op):
	""" Test operation raises exception for point empty data """
	data = numpy.zeros((0,2))
	caller = callerCon(data)
	callee = calleeConstructor(data, calleeCon)

	toCall = getattr(caller, op)
	toCall(callee)

def back_fEmptyException(callerCon, calleeCon, op):
	""" Test operation raises exception for feature empty data """
	data = [[],[]]
	caller = callerCon(data)
	callee = calleeConstructor(data, calleeCon)

	toCall = getattr(caller, op)
	toCall(callee)

def back_byZeroException(callerCon, calleeCon, op):
	""" Test operation when other data contains zero """
	data1 = [[1,2,6], [4,5,3], [7,8,6]]
	data2 = [[1,2,3],[0,0,0],[6,7,8]]
	caller = callerCon(data1)
	callee = calleeConstructor(data2, calleeCon)

	toCall = getattr(caller, op)
	toCall(callee)

def back_byInfException(callerCon, calleeCon, op):
	""" Test operation when other data contains an infinity """
	data1 = [[1,2,6], [4,5,3], [7,8,6]]
	data2 = [[1,2,3],[5,numpy.Inf,10],[6,7,8]]
	caller = callerCon(data1)
	callee = calleeConstructor(data2, calleeCon)

	toCall = getattr(caller, op)
	toCall(callee)

def makeAllData(constructor, rhsCons, n, sparsity):
	randomlf = UML.createRandomData('Matrix', n, n, sparsity)
	randomrf = UML.createRandomData('Matrix', n, n, sparsity)
	lhsf = randomlf.copyAs("numpymatrix")
	rhsf = randomrf.copyAs("numpymatrix")
	lhsi = numpy.matrix(numpyRandom.random_integers(1,10,(n,n)), dtype=float)
	rhsi = numpy.matrix(numpyRandom.random_integers(1,10,(n,n)), dtype=float)
		
	lhsfObj = constructor(lhsf)
	lhsiObj = constructor(lhsi)
	rhsfObj = None
	rhsiObj = None
	if rhsCons is not None:
		rhsfObj = rhsCons(rhsf)
		rhsiObj = rhsCons(rhsi)

	return (lhsf,rhsf,lhsi,rhsi,lhsfObj,rhsfObj,lhsiObj,rhsiObj)


def back_autoVsNumpyObjCallee(constructor, npOp, UMLOp, UMLinplace, sparsity):
	""" Test operation of automated data against numpy operations """
	trials = 1
	for t in range(trials):
		n = pythonRandom.randint(1,15)

		datas = makeAllData(constructor,constructor,n, sparsity)	
		(lhsf,rhsf,lhsi,rhsi,lhsfObj,rhsfObj,lhsiObj,rhsiObj) = datas

		resultf = npOp(lhsf, rhsf)
		resulti = npOp(lhsi, rhsi)
		resfObj = getattr(lhsfObj, UMLOp)(rhsfObj)
		resiObj = getattr(lhsiObj, UMLOp)(rhsiObj)

		expfObj = constructor(resultf)
		expiObj = constructor(resulti)

		if UMLinplace:
			assert expfObj.isApproximatelyEqual(lhsfObj)
			assert expiObj.isIdentical(lhsiObj)
		else:
			assert expfObj.isApproximatelyEqual(resfObj)
			assert expiObj.isIdentical(resiObj)

def back_autoVsNumpyScalar(constructor, npOp, UMLOp, UMLinplace, sparsity):
	""" Test operation of automated data with a scalar argument, against numpy operations """
	lside = UMLOp.startswith('__r')
	trials = 5
	for t in range(trials):
		n = pythonRandom.randint(1,10)

		scalar = pythonRandom.randint(1,4)

		datas = makeAllData(constructor,None,n, sparsity)	
		(lhsf,rhsf,lhsi,rhsi,lhsfObj,rhsfObj,lhsiObj,rhsiObj) = datas

		if lside:
			resultf = npOp(scalar, lhsf)
			resulti = npOp(scalar, lhsi)
			resfObj = getattr(lhsfObj, UMLOp)(scalar)
			resiObj = getattr(lhsiObj, UMLOp)(scalar)
		else:
			resultf = npOp(lhsf, scalar)
			resulti = npOp(lhsi, scalar)
			resfObj = getattr(lhsfObj, UMLOp)(scalar)
			resiObj = getattr(lhsiObj, UMLOp)(scalar)

		expfObj = constructor(resultf)
		expiObj = constructor(resulti)

		if UMLinplace:
			assert expfObj.isApproximatelyEqual(lhsfObj)
			assert expiObj.isIdentical(lhsiObj)
		else:
			assert expfObj.isApproximatelyEqual(resfObj)
			assert expiObj.isIdentical(resiObj)

def back_autoVsNumpyObjCalleeDiffTypes(constructor, npOp, UMLOp, UMLinplace, sparsity):
	""" Test operation on handmade data with different types of data objects"""
	makers = [UML.data.List, UML.data.Matrix, UML.data.Sparse]
	
	for i in range(len(makers)):
		maker = makers[i]
		n = pythonRandom.randint(1,10)

		datas = makeAllData(constructor,maker,n, sparsity)	
		(lhsf,rhsf,lhsi,rhsi,lhsfObj,rhsfObj,lhsiObj,rhsiObj) = datas

		resultf = npOp(lhsf, rhsf)
		resulti = npOp(lhsi, rhsi)
		resfObj = getattr(lhsfObj, UMLOp)(rhsfObj)
		resiObj = getattr(lhsiObj, UMLOp)(rhsiObj)

		expfObj = constructor(resultf)
		expiObj = constructor(resulti)

		if UMLinplace:
			assert expfObj.isApproximatelyEqual(lhsfObj)
			assert expiObj.isIdentical(lhsiObj)
		else:
			assert expfObj.isApproximatelyEqual(resfObj)
			assert expiObj.isIdentical(resiObj)

			if type(resfObj) != type(lhsfObj):
				assert type(resfObj) == maker.__class__
			if type(resiObj) != type(lhsiObj):
				assert type(resfObj) == maker.__class__

def wrapAndCall(toWrap, expected, *args):
	try:
		toWrap(*args)
	except expected:
		pass
	except:
		raise

def run_full_backendDivMod(constructor, npEquiv, UMLOp, inplace, sparsity):
	wrapAndCall(back_byZeroException, ZeroDivisionError, *(constructor, constructor, UMLOp))
	wrapAndCall(back_byInfException, ArgumentException, *(constructor, constructor, UMLOp))

	run_full_backend(constructor, npEquiv, UMLOp, inplace, sparsity)

def run_full_backend(constructor, npEquiv, UMLOp, inplace, sparsity):
	wrapAndCall(back_otherObjectExceptions, ArgumentException, *(constructor, UMLOp))

	wrapAndCall(back_selfNotNumericException, ArgumentException, *(constructor, constructor, UMLOp))

	wrapAndCall(back_otherNotNumericException, ArgumentException, *(constructor, constructor, UMLOp))

	wrapAndCall(back_pShapeException, ArgumentException, *(constructor, constructor, UMLOp))

	wrapAndCall(back_fShapeException, ArgumentException, *(constructor, constructor, UMLOp))

	wrapAndCall(back_pEmptyException, ImproperActionException, *(constructor, constructor, UMLOp))

	wrapAndCall(back_fEmptyException, ImproperActionException, *(constructor, constructor, UMLOp))

	back_autoVsNumpyObjCallee(constructor, npEquiv, UMLOp, inplace, sparsity)

	back_autoVsNumpyScalar(constructor, npEquiv, UMLOp, inplace, sparsity)

	back_autoVsNumpyObjCalleeDiffTypes(constructor, npEquiv, UMLOp, inplace, sparsity)

def run_full_backendDivMod_rop(constructor, npEquiv, UMLOp, inplace, sparsity):
	run_full_backend_rOp(constructor, npEquiv, UMLOp, inplace, sparsity)

def run_full_backend_rOp(constructor, npEquiv, UMLOp, inplace, sparsity):
	wrapAndCall(back_otherObjectExceptions, ArgumentException, *(constructor, UMLOp))

	wrapAndCall(back_selfNotNumericException, ArgumentException, *(constructor, constructor, UMLOp))

	wrapAndCall(back_pEmptyException, ImproperActionException, *(constructor, constructor, UMLOp))

	wrapAndCall(back_fEmptyException, ImproperActionException, *(constructor, constructor, UMLOp))

	back_autoVsNumpyScalar(constructor, npEquiv, UMLOp, inplace, sparsity)



class NumericalBackend(DataTestObject):

	#############################
	# elementwiseMultiply #
	#############################

	@raises(ArgumentException)
	def test_elementwiseMultiply_otherObjectExceptions(self):
		""" Test elementwiseMultiply raises exception when param is not a UML data object """
		back_otherObjectExceptions(self.constructor, 'elementwiseMultiply')

	@raises(ArgumentException)
	def test_elementwiseMultiply_selfNotNumericException(self):
		""" Test elementwiseMultiply raises exception if self has non numeric data """
		back_selfNotNumericException(self.constructor, self.constructor, 'elementwiseMultiply')

	@raises(ArgumentException)
	def test_elementwiseMultiply_otherNotNumericException(self):
		""" Test elementwiseMultiply raises exception if param object has non numeric data """
		back_otherNotNumericException(self.constructor, self.constructor, 'elementwiseMultiply')

	@raises(ArgumentException)
	def test_elementwiseMultiply_pShapeException(self):
		""" Test elementwiseMultiply raises exception the shapes of the object don't fit correctly """
		back_pShapeException(self.constructor, self.constructor, 'elementwiseMultiply')

	@raises(ArgumentException)
	def test_elementwiseMultiply_fShapeException(self):
		""" Test elementwiseMultiply raises exception the shapes of the object don't fit correctly """
		back_fShapeException(self.constructor, self.constructor, 'elementwiseMultiply')

	@raises(ImproperActionException)
	def test_elementwiseMultiply_pEmptyException(self):
		""" Test elementwiseMultiply raises exception for point empty data """
		back_pEmptyException(self.constructor, self.constructor, 'elementwiseMultiply')

	@raises(ImproperActionException)
	def test_elementwiseMultiply_fEmptyException(self):
		""" Test elementwiseMultiply raises exception for feature empty data """
		back_fEmptyException(self.constructor, self.constructor, 'elementwiseMultiply')

	def test_elementwiseMultiply_handmade(self):
		""" Test elementwiseMultiply on handmade data """
		data = [[1,2], [4,5], [7,8]]
		twos = [[2,2], [2,2], [2,2]]
		exp1 = [[2,4], [8,10], [14,16]]
		halves = [[0.5,0.5], [0.5,0.5], [0.5,0.5]]

		caller = self.constructor(data)
		twosObj = self.constructor(twos)
		caller.elementwiseMultiply(twosObj)

		exp1Obj = self.constructor(exp1)

		assert exp1Obj.isIdentical(caller)

		halvesObj = self.constructor(halves)
		caller.elementwiseMultiply(halvesObj)

		exp2Obj = self.constructor(data)

		assert caller.isIdentical(exp2Obj)

	def test_elementwiseMultiply_handmadeDifInputs(self):
		""" Test elementwiseMultiply on handmade data with different input object types"""
		data = [[1,2], [4,5], [7,8]]
		twos = [[2,2], [2,2], [2,2]]
		exp1 = [[2,4], [8,10], [14,16]]
		halves = [[0.5,0.5], [0.5,0.5], [0.5,0.5]]

		makers = [UML.data.List, UML.data.Matrix, UML.data.Sparse]

		for maker in makers:
			caller = self.constructor(data)
			twosObj = maker(twos)
			caller.elementwiseMultiply(twosObj)

			exp1Obj = self.constructor(exp1)

			assert exp1Obj.isIdentical(caller)

			halvesObj = maker(halves)
			caller.elementwiseMultiply(halvesObj)

			exp2Obj = self.constructor(data)

			assert caller.isIdentical(exp2Obj)

	def test_elementwiseMultipy_auto(self):
		""" Test elementwiseMultiply on generated data against the numpy op """
		makers = [UML.data.List, UML.data.Matrix, UML.data.Sparse]
	
		for i in range(len(makers)):
			maker = makers[i]
			n = pythonRandom.randint(1,10)

			randomlf = UML.createRandomData('Matrix', n, n, .2)
			randomrf = UML.createRandomData('Matrix', n, n, .2)
			lhsf = randomlf.copyAs("numpymatrix")
			rhsf = randomrf.copyAs("numpymatrix")
			lhsi = numpy.matrix(numpy.ones((n,n)))
			rhsi = numpy.matrix(numpy.ones((n,n)))
			
			lhsfObj = self.constructor(lhsf)
			rhsfObj = maker(rhsf)
			lhsiObj = self.constructor(lhsi)
			rhsiObj = maker(rhsi)

			resultf = numpy.multiply(lhsf, rhsf)
			resulti = numpy.multiply(lhsi, rhsi)
			lhsfObj.elementwiseMultiply(rhsfObj)
			lhsiObj.elementwiseMultiply(rhsiObj)

			expfObj = self.constructor(resultf)
			expiObj = self.constructor(resulti)

			assert expfObj.isApproximatelyEqual(lhsfObj)
			assert expiObj.isIdentical(lhsiObj)
			
	###########
	# __mul__ #
	###########

	@raises(ArgumentException)
	def test_mul_selfNotNumericException(self):
		""" Test __mul__ raises exception if self has non numeric data """
		back_selfNotNumericException(self.constructor, self.constructor, '__mul__')

	@raises(ArgumentException)
	def test_mul_otherNotNumericException(self):
		""" Test __mul__ raises exception if param object has non numeric data """
		back_otherNotNumericException(self.constructor, self.constructor, '__mul__')

	@raises(ArgumentException)
	def test_mul_shapeException(self):
		""" Test __mul__ raises exception the shapes of the object don't fit correctly """
		data1 = [[1,2], [4,5], [7,8]]
		data2 = [[1,2,3], [4,5,6], [7,8,9]]
		caller = self.constructor(data1)
		callee = self.constructor(data2)

		caller * callee

	@raises(ImproperActionException)
	def test_mul_pEmptyException(self):
		""" Test __mul__ raises exception for point empty data """
		data = []
		fnames = ['one', 'two']
		caller = self.constructor(data, featureNames=fnames)
		callee = caller.copy()
		callee.transpose()

		caller * callee

	@raises(ImproperActionException)
	def test_mul_fEmptyException(self):
		""" Test __mul__ raises exception for feature empty data """
		data = [[],[]]
		pnames = ['one', 'two']
		caller = self.constructor(data, pointNames=pnames)
		callee = caller.copy()
		callee.transpose()

		caller * callee

	def test_mul_autoObjs(self):
		""" Test __mul__ against automated data """
		back_autoVsNumpyObjCallee(self.constructor, numpy.dot, '__mul__', False, 0.2)

	def test_mul_autoScalar(self):
		""" Test __mul__ of a scalar against automated data """
		back_autoVsNumpyScalar(self.constructor, numpy.dot, '__mul__', False, 0.2)

	def test_autoVsNumpyObjCalleeDiffTypes(self):
		""" Test __mul__ against generated data with different UML types of objects """
		back_autoVsNumpyObjCalleeDiffTypes(self.constructor, numpy.dot, '__mul__', False, 0.2)

	############
	# __rmul__ #
	############

	def test_rmul_autoScalar(self):
		""" Test __rmul__ of a scalar against automated data """
		back_autoVsNumpyScalar(self.constructor, numpy.multiply, '__rmul__', False, 0.2)


	############
	# __imul__ #
	############

	@raises(ArgumentException)
	def test_imul_selfNotNumericException(self):
		""" Test __imul__ raises exception if self has non numeric data """
		back_selfNotNumericException(self.constructor, self.constructor, '__imul__')

	@raises(ArgumentException)
	def test_imul_otherNotNumericException(self):
		""" Test __imul__ raises exception if param object has non numeric data """
		back_otherNotNumericException(self.constructor, self.constructor, '__imul__')

	@raises(ArgumentException)
	def test_imul_shapeException(self):
		""" Test __imul__ raises exception the shapes of the object don't fit correctly """
		data1 = [[1,2], [4,5], [7,8]]
		data2 = [[1,2,3], [4,5,6], [7,8,9]]
		caller = self.constructor(data1)
		callee = self.constructor(data2)

		caller.__imul__(callee)

	@raises(ImproperActionException)
	def test_imul_pEmptyException(self):
		""" Test __imul__ raises exception for point empty data """
		data = []
		fnames = ['one', 'two']
		caller = self.constructor(data, featureNames=fnames)
		callee = caller.copy()
		callee.transpose()

		caller *= callee

	@raises(ImproperActionException)
	def test_imul_fEmptyException(self):
		""" Test __imul__ raises exception for feature empty data """
		data = [[],[]]
		pnames = ['one', 'two']
		caller = self.constructor(data, pointNames=pnames)
		callee = caller.copy()
		callee.transpose()

		caller *= callee

	def test_imul_autoObjs(self):
		""" Test __imul__ against automated data """
		back_autoVsNumpyObjCallee(self.constructor, numpy.dot,'__imul__', True, 0.2)

	def test_imul_autoScalar(self):
		""" Test __imul__ of a scalar against automated data """
		back_autoVsNumpyScalar(self.constructor, numpy.dot, '__imul__', True, 0.2)

	def test_imul__autoVsNumpyObjCalleeDiffTypes(self):
		""" Test __mul__ against generated data with different UML types of objects """
		back_autoVsNumpyObjCalleeDiffTypes(self.constructor, numpy.dot, '__mul__', False, 0.2)


	############
	# __add__ #
	############

	def test_add_fullSuite(self):
		""" __add__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend(self.constructor, numpy.add, '__add__', False, 0.2)

	############
	# __radd__ #
	############

	def test_radd_fullSuite(self):
		""" __radd__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend_rOp(self.constructor, numpy.add, '__radd__', False, 0.2)

	############
	# __iadd__ #
	############

	def test_iadd_fullSuite(self):
		""" __iadd__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend(self.constructor, numpy.add, '__iadd__', True, 0.2)

	############
	# __sub__ #
	############

	def test_sub_fullSuite(self):
		""" __sub__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend(self.constructor, numpy.subtract, '__sub__', False, 0.2)

	############
	# __rsub__ #
	############

	def test_rsub_fullSuite(self):
		""" __rsub__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend_rOp(self.constructor, numpy.subtract, '__rsub__', False, 0.2)

	############
	# __isub__ #
	############

	def test_isub_fullSuite(self):
		""" __isub__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend(self.constructor, numpy.subtract, '__isub__', True, 0.2)

	############
	# __div__ #
	############

	def test_div_fullSuite(self):
		""" __div__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.divide, '__div__', False, 0)

	############
	# __rdiv__ #
	############

	def test_rdiv_fullSuite(self):
		""" __rdiv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod_rop(self.constructor, numpy.divide, '__rdiv__', False, 0)

	############
	# __idiv__ #
	############

	def test_idiv_fullSuite(self):
		""" __idiv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.divide, '__idiv__', True, 0)


	###############
	# __truediv__ #
	###############

	def test_truediv_fullSuite(self):
		""" __truediv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.true_divide, '__truediv__', False, 0)

	################
	# __rtruediv__ #
	################

	def test_rtruediv_fullSuite(self):
		""" __rtruediv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod_rop(self.constructor, numpy.true_divide, '__rtruediv__', False, 0)

	################
	# __itruediv__ #
	################

	def test_itruediv_fullSuite(self):
		""" __itruediv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.true_divide, '__itruediv__', True, 0)


	###############
	# __floordiv__ #
	###############

	def test_floordiv_fullSuite(self):
		""" __floordiv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.floor_divide, '__floordiv__', False, 0)

	################
	# __rfloordiv__ #
	################

	def test_rfloordiv_fullSuite(self):
		""" __rfloordiv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod_rop(self.constructor, numpy.floor_divide, '__rfloordiv__', False, 0)

	################
	# __ifloordiv__ #
	################

	def test_ifloordiv_fullSuite(self):
		""" __ifloordiv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.floor_divide, '__ifloordiv__', True, 0)

	###############
	# __mod__ #
	###############

	def test_mod_fullSuite(self):
		""" __mod__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.mod, '__mod__', False, 0)

	################
	# __rmod__ #
	################

	def test_rmod_fullSuite(self):
		""" __rmod__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod_rop(self.constructor, numpy.mod, '__rmod__', False, 0)

	################
	# __imod__ #
	################

	def test_imod_fullSuite(self):
		""" __imod__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.mod, '__imod__', True, 0)



	###########
	# __pow__ #
	###########

	def test_pow_exceptions(self):
		""" __pow__ Run the full standardized suite of tests for a binary numeric op """
		constructor = self.constructor
		UMLOp = '__pow__'
		inputs = (constructor, UMLOp)
		wrapAndCall(back_otherObjectExceptions, ArgumentException, *inputs)

		inputs = (constructor, int, UMLOp)
		wrapAndCall(back_selfNotNumericException, ArgumentException, *inputs)

		inputs = (constructor, constructor, UMLOp)
		wrapAndCall(back_pEmptyException, ImproperActionException, *inputs)

		inputs = (constructor, constructor, UMLOp)
		wrapAndCall(back_fEmptyException, ImproperActionException, *inputs)

	def test_pow_autoVsNumpyScalar(self):
		""" Test __pow__ with automated data and a scalar argument, against numpy operations """
		trials = 5
		for t in range(trials):
			n = pythonRandom.randint(1,15)
			scalar = pythonRandom.randint(0,5)

			datas = makeAllData(self.constructor,None,n, .02)	
			(lhsf,rhsf,lhsi,rhsi,lhsfObj,rhsfObj,lhsiObj,rhsiObj) = datas

			resultf = lhsf ** scalar
			resulti = lhsi ** scalar
			resfObj = lhsfObj **scalar
			resiObj = lhsiObj **scalar

			expfObj = self.constructor(resultf)
			expiObj = self.constructor(resulti)

			assert expfObj.isApproximatelyEqual(resfObj)
			assert expiObj.isIdentical(resiObj)


	###########
	# __ipow__ #
	###########

	def test_ipow_exceptions(self):
		""" __ipow__ Run the full standardized suite of tests for a binary numeric op """
		constructor = self.constructor
		UMLOp = '__ipow__'
		inputs = (constructor, UMLOp)
		wrapAndCall(back_otherObjectExceptions, ArgumentException, *inputs)

		inputs = (constructor, int, UMLOp)
		wrapAndCall(back_selfNotNumericException, ArgumentException, *inputs)

		inputs = (constructor, constructor, UMLOp)
		wrapAndCall(back_pEmptyException, ImproperActionException, *inputs)

		inputs = (constructor, constructor, UMLOp)
		wrapAndCall(back_fEmptyException, ImproperActionException, *inputs)

	def test_ipow_autoVsNumpyScalar(self):
		""" Test __ipow__ with automated data and a scalar argument, against numpy operations """
		trials = 5
		for t in range(trials):
			n = pythonRandom.randint(1,15)
			scalar = pythonRandom.randint(0,5)

			datas = makeAllData(self.constructor,None,n, .02)	
			(lhsf,rhsf,lhsi,rhsi,lhsfObj,rhsfObj,lhsiObj,rhsiObj) = datas

			resultf = lhsf ** scalar
			resulti = lhsi ** scalar
			resfObj = lhsfObj.__ipow__(scalar)
			resiObj = lhsiObj.__ipow__(scalar)

			expfObj = self.constructor(resultf)
			expiObj = self.constructor(resulti)

			assert expfObj.isApproximatelyEqual(resfObj)
			assert expiObj.isIdentical(resiObj)
			assert resfObj.isIdentical(lhsfObj)
			assert resiObj.isIdentical(lhsiObj)



	###########
	# __pos__ #
	###########

	def test_pos_DoesntCrash(self):
		""" Test that __pos__ does nothing and doesn't crash """
		data1 = [[1,2], [4,5], [7,8]]
		caller = self.constructor(data1)

		ret1 = +caller
		ret2 = caller.__pos__()

		assert caller.isIdentical(ret1)
		assert caller.isIdentical(ret2)

	###########
	# __neg__ #
	###########

	def test_neg_simple(self):
		""" Test that __neg__ works as expected on some simple data """
		data1 = [[1,2], [-4,-5], [7,-8], [0,0]]
		data2 = [[-1,-2], [4,5], [-7,8], [0,0]]
		caller = self.constructor(data1)
		exp = self.constructor(data2)

		ret1 = -caller
		ret2 = caller.__neg__()

		assert exp.isIdentical(ret1)
		assert exp.isIdentical(ret2)

	###########
	# __abs__ #
	###########

	def test_abs_simple(self):
		""" Test that __abs__ works as expected on some simple data """
		data1 = [[1,2], [-4,-5], [7,-8], [0,0]]
		data2 = [[1,2], [4,5], [7,8], [0,0]]
		caller = self.constructor(data1)
		exp = self.constructor(data2)

		ret1 = abs(caller)
		ret2 = caller.__abs__()

		assert exp.isIdentical(ret1)
		assert exp.isIdentical(ret2)
	
	
