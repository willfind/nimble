
import sys
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


def back_unary_name_preservations(callerCon, op):
	""" Test that point / feature names are preserved when calling a unary op """
	data = [[1,1,1], [1,1,1], [1,1,1]]
	pnames = {'p1':0, 'p2':1, 'p3':2}
	fnames = {'f1':0, 'f2':1, 'f3':2}

	caller = callerCon(data, pnames, fnames)
	toCall = getattr(caller, op)
	ret = toCall()

	assert ret.pointNames == pnames
	assert ret.featureNames == fnames

	caller.setPointName('p1', 'p0')

	assert 'p1' in ret.pointNames
	assert 'p0' not in ret.pointNames
	assert 'p0' in caller.pointNames
	assert 'p1' not in caller.pointNames

def back_binaryscalar_name_preservations(callerCon, op, inplace):
	""" Test that p/f names are preserved when calling a binary scalar op """
	data = [[1,1,1], [1,1,1], [1,1,1]]
	pnames = {'p1':0, 'p2':1, 'p3':2}
	fnames = {'f1':0, 'f2':1, 'f3':2}

	for num in [-2, 0, 1, 4]:
		try:
			caller = callerCon(data, pnames, fnames)
			toCall = getattr(caller, op)
			ret = toCall(num)

			assert ret.pointNames == pnames
			assert ret.featureNames == fnames

			caller.setPointName('p1', 'p0')
			if inplace:
				assert 'p0' in ret.pointNames
				assert 'p1' not in ret.pointNames
			else:
				assert 'p0' not in ret.pointNames
				assert 'p1' in ret.pointNames
			assert 'p0' in caller.pointNames
			assert 'p1' not in caller.pointNames
		except AssertionError:
			einfo = sys.exc_info()
			raise einfo[1], None, einfo[2]
#		except ArgumentException:
#			einfo = sys.exc_info()
#			raise einfo[1], None, einfo[2]
		except:
			pass

def back_binaryelementwise_name_preservations(callerCon, op, inplace):
	""" Test that p/f names are preserved when calling a binary element wise op """
	data = [[1,1,1], [1,1,1], [1,1,1]]
	pnames = {'p1':0, 'p2':1, 'p3':2}
	fnames = {'f1':0, 'f2':1, 'f3':2}
	
	otherRaw = [[1,1,1], [1,1,1], [1,1,1]]

	# names not the same
	caller = callerCon(data, pnames, fnames)
	opnames = pnames
	ofnames = {'f0':0, 'f1':1, 'f2':2}
	other = callerCon(otherRaw, opnames, ofnames)
	try:
		toCall = getattr(caller, op)
		ret = toCall(other)
		if ret != NotImplemented:
			assert False
	except ArgumentException:
		pass
	# if it isn't the exception we expect, pass it on
	except:
		einfo = sys.exc_info()
		raise einfo[1], None, einfo[2]

	# names interwoven
	other = callerCon(otherRaw, pnames, None)
	caller = callerCon(data, None, fnames)
	toCall = getattr(caller, op)
	ret = toCall(other)

	if ret != NotImplemented:
		assert ret.pointNames == pnames
		assert ret.featureNames == fnames

	# both names same
	caller = callerCon(data, pnames, fnames)
	other = callerCon(otherRaw, pnames, fnames)
	toCall = getattr(caller, op)
	ret = toCall(other)

	if ret != NotImplemented:
		assert ret.pointNames == pnames
		assert ret.featureNames == fnames

		caller.setPointName('p1', 'p0')
		if inplace:
			assert 'p0' in ret.pointNames
			assert 'p1' not in ret.pointNames
		else:
			assert 'p0' not in ret.pointNames
			assert 'p1' in ret.pointNames
		assert 'p0' in caller.pointNames
		assert 'p1' not in caller.pointNames

def back_matrixmul_name_preservations(callerCon, op, inplace):
	""" Test that p/f names are preserved when calling a binary element wise op """
	data = [[1,1,1], [1,1,1], [1,1,1]]
	pnames = {'p1':0, 'p2':1, 'p3':2}
	fnames = {'f1':0, 'f2':1, 'f3':2}
	
	# [p x f1] time [f2 xp] where f1 != f2
	caller = callerCon(data, pnames, fnames)
	ofnames = {'f0':0, 'f1':1, 'f2':2}
	other = callerCon(data, ofnames, pnames)
	try:
		toCall = getattr(caller, op)
		ret = toCall(other)
		if ret != NotImplemented:
			assert False
	except ArgumentException:
		pass
	# if it isn't the exception we expect, pass it on
	except:
		einfo = sys.exc_info()
		raise einfo[1], None, einfo[2]

	# names interwoven
	caller = callerCon(data, pnames, fnames)
	other = callerCon(data, None, fnames)
	other.setPointName(0, 'f1')
	other.setPointName(1, 'f2')
	toCall = getattr(caller, op)
	ret = toCall(other)

	if ret != NotImplemented:
		assert ret.pointNames == pnames
		assert ret.featureNames == fnames

	# both names same
	caller = callerCon(data, pnames, pnames)
	other = callerCon(data, pnames, fnames)
	toCall = getattr(caller, op)
	ret = toCall(other)

	if ret != NotImplemented:
		assert ret.pointNames == pnames
		assert ret.featureNames == fnames

		caller.setPointName('p1', 'p0')
		if inplace:
			assert 'p0' in ret.pointNames
			assert 'p1' not in ret.pointNames
		else:
			assert 'p0' not in ret.pointNames
			assert 'p1' in ret.pointNames
		assert 'p0' in caller.pointNames
		assert 'p1' not in caller.pointNames

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

	####################
	# elementwisePower #
	####################

	@raises(ArgumentException)
	def test_elementwisePower_otherObjectExceptions(self):
		""" Test elementwiseMultiply raises exception when param is not a UML data object """
		back_otherObjectExceptions(self.constructor, 'elementwisePower')

	@raises(ArgumentException)
	def test_elementwisePower_selfNotNumericException(self):
		""" Test elementwisePower raises exception if self has non numeric data """
		back_selfNotNumericException(self.constructor, self.constructor, 'elementwisePower')

	@raises(ArgumentException)
	def test_elementwisePower_otherNotNumericException(self):
		""" Test elementwisePower raises exception if param object has non numeric data """
		back_otherNotNumericException(self.constructor, self.constructor, 'elementwisePower')

	@raises(ArgumentException)
	def test_elementwisePower_pShapeException(self):
		""" Test elementwisePower raises exception the shapes of the object don't fit correctly """
		back_pShapeException(self.constructor, self.constructor, 'elementwisePower')

	@raises(ArgumentException)
	def test_elementwisePower_fShapeException(self):
		""" Test elementwisePower raises exception the shapes of the object don't fit correctly """
		back_fShapeException(self.constructor, self.constructor, 'elementwisePower')

	@raises(ImproperActionException)
	def test_elementwisePower_pEmptyException(self):
		""" Test elementwisePower raises exception for point empty data """
		back_pEmptyException(self.constructor, self.constructor, 'elementwisePower')

	@raises(ImproperActionException)
	def test_elementwisePower_fEmptyException(self):
		""" Test elementwisePower raises exception for feature empty data """
		back_fEmptyException(self.constructor, self.constructor, 'elementwisePower')

	def test_elementwisePower_handmade(self):
		""" Test elementwisePower on handmade data """
		data = [[1.0,2], [4,5], [7,4]]
		exponents = [[0,-1], [-.5,2], [2,.5]]
		exp1 = [[1,.5], [.5,25], [49,2]]
		callerpnames = ['1', '2', '3']

		calleepnames = ['I', 'dont', 'match']
		calleefnames = ['one', 'two']

		makers = [UML.data.List, UML.data.Matrix, UML.data.Sparse]
		for maker in makers:
			caller = self.constructor(data, pointNames=callerpnames)
			exponentsObj = maker(exponents, pointNames=calleepnames, featureNames=calleefnames)
			caller.elementwisePower(exponentsObj)

			exp1Obj = self.constructor(exp1, pointNames=callerpnames)

			assert exp1Obj.isIdentical(caller)


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
	
	def test_elementwiseMultipy_name_preservations(self):
		""" Test p/f names are preserved when calling elementwiseMultipy"""
		data = [[1,1,1], [1,1,1], [1,1,1]]
		pnames = {'p1':0, 'p2':1, 'p3':2}
		fnames = {'f1':0, 'f2':1, 'f3':2}
		
		otherRaw = [[1,1,1], [1,1,1], [1,1,1]]

		# names not the same
		caller = self.constructor(data, pnames, fnames)
		opnames = pnames
		ofnames = {'f0':0, 'f1':1, 'f2':2}
		other = self.constructor(otherRaw, opnames, ofnames)
		try:
			toCall = getattr(caller, 'elementwiseMultiply')
			ret = toCall(other)
			assert False
		except ArgumentException:
			pass
		# if it isn't the exception we expect, pass it on
		except:
			einfo = sys.exc_info()
			raise einfo[1], None, einfo[2]

		# names interwoven
		other = self.constructor(otherRaw, pnames, None)
		caller = self.constructor(data, None, fnames)
		toCall = getattr(caller, 'elementwiseMultiply')
		ret = toCall(other)

		assert ret is None
		assert caller.pointNames == pnames
		assert caller.featureNames == fnames

		# both names same
		caller = self.constructor(data, pnames, fnames)
		other = self.constructor(otherRaw, pnames, fnames)
		toCall = getattr(caller, 'elementwiseMultiply')
		ret = toCall(other)

		assert caller.pointNames == pnames
		assert caller.featureNames == fnames

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

	def test_mul_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __mul__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__mul__', False)

	def test_mul_matrixmul_name_preservations(self):
		""" Test p/f names are preserved when calling __mul__ with obj arg"""
		back_matrixmul_name_preservations(self.constructor, '__mul__', False)

	############
	# __rmul__ #
	############

	def test_rmul_autoScalar(self):
		""" Test __rmul__ of a scalar against automated data """
		back_autoVsNumpyScalar(self.constructor, numpy.multiply, '__rmul__', False, 0.2)

	def test_rmul_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __rmul__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__rmul__', False)

	def test_rmul_matrixmul_name_preservations(self):
		""" Test p/f names are preserved when calling __rmul__ with obj arg"""
		back_matrixmul_name_preservations(self.constructor, '__rmul__', False)


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

	def test_imul_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __imul__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__imul__', True)

	def test_imul_matrixmul_name_preservations(self):
		""" Test p/f names are preserved when calling __imul__ with obj arg"""
		back_matrixmul_name_preservations(self.constructor, '__imul__', True)


	############
	# __add__ #
	############

	def test_add_fullSuite(self):
		""" __add__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend(self.constructor, numpy.add, '__add__', False, 0.2)

	def test_add_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __add__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__add__', False)

	def test_add_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __add__"""
		back_binaryelementwise_name_preservations(self.constructor, '__add__', False)


	############
	# __radd__ #
	############

	def test_radd_fullSuite(self):
		""" __radd__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend_rOp(self.constructor, numpy.add, '__radd__', False, 0.2)

	def test_radd_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __radd__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__radd__', False)

	def test_radd_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __radd__"""
		back_binaryelementwise_name_preservations(self.constructor, '__radd__', False)


	############
	# __iadd__ #
	############

	def test_iadd_fullSuite(self):
		""" __iadd__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend(self.constructor, numpy.add, '__iadd__', True, 0.2)

	def test_iadd_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __iadd__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__iadd__', True)

	def test_iadd_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __iadd__"""
		back_binaryelementwise_name_preservations(self.constructor, '__iadd__', True)


	############
	# __sub__ #
	############

	def test_sub_fullSuite(self):
		""" __sub__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend(self.constructor, numpy.subtract, '__sub__', False, 0.2)

	def test_sub_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __sub__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__sub__', False)

	def test_sub_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __sub__"""
		back_binaryelementwise_name_preservations(self.constructor, '__sub__', False)


	############
	# __rsub__ #
	############

	def test_rsub_fullSuite(self):
		""" __rsub__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend_rOp(self.constructor, numpy.subtract, '__rsub__', False, 0.2)

	def test_rsub_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __rsub__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__rsub__', False)

	def test_rsub_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __rsub__"""
		back_binaryelementwise_name_preservations(self.constructor, '__rsub__', False)


	############
	# __isub__ #
	############

	def test_isub_fullSuite(self):
		""" __isub__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backend(self.constructor, numpy.subtract, '__isub__', True, 0.2)

	def test_isub_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __isub__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__isub__', True)

	def test_isub_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __isub__"""
		back_binaryelementwise_name_preservations(self.constructor, '__isub__', True)


	############
	# __div__ #
	############

	def test_div_fullSuite(self):
		""" __div__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.divide, '__div__', False, 0)

	def test_div_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __div__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__div__', False)

	def test_div_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __div__"""
		back_binaryelementwise_name_preservations(self.constructor, '__div__', False)


	############
	# __rdiv__ #
	############

	def test_rdiv_fullSuite(self):
		""" __rdiv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod_rop(self.constructor, numpy.divide, '__rdiv__', False, 0)

	def test_rdiv_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __rdiv__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__rdiv__', False)

	def test_rdiv_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __rdiv__"""
		back_binaryelementwise_name_preservations(self.constructor, '__rdiv__', False)


	############
	# __idiv__ #
	############

	def test_idiv_fullSuite(self):
		""" __idiv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.divide, '__idiv__', True, 0)

	def test_idiv_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __idiv__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__idiv__', True)

	def test_idiv_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __idiv__"""
		back_binaryelementwise_name_preservations(self.constructor, '__idiv__', True)


	###############
	# __truediv__ #
	###############

	def test_truediv_fullSuite(self):
		""" __truediv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.true_divide, '__truediv__', False, 0)

	def test_truediv_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __truediv__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__truediv__', False)

	def test_truediv_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __truediv__"""
		back_binaryelementwise_name_preservations(self.constructor, '__truediv__', False)


	################
	# __rtruediv__ #
	################

	def test_rtruediv_fullSuite(self):
		""" __rtruediv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod_rop(self.constructor, numpy.true_divide, '__rtruediv__', False, 0)

	def test_rtruediv_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __rtruediv__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__rtruediv__', False)

	def test_rtruediv_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __rtruediv__"""
		back_binaryelementwise_name_preservations(self.constructor, '__rtruediv__', False)


	################
	# __itruediv__ #
	################

	def test_itruediv_fullSuite(self):
		""" __itruediv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.true_divide, '__itruediv__', True, 0)

	def test_itruediv_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __itruediv__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__itruediv__', True)

	def test_itruediv_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __itruediv__"""
		back_binaryelementwise_name_preservations(self.constructor, '__itruediv__', True)


	###############
	# __floordiv__ #
	###############

	def test_floordiv_fullSuite(self):
		""" __floordiv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.floor_divide, '__floordiv__', False, 0)

	def test_floordiv_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __floordiv__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__floordiv__', False)

	def test_floordiv_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __floordiv__"""
		back_binaryelementwise_name_preservations(self.constructor, '__floordiv__', False)


	################
	# __rfloordiv__ #
	################

	def test_rfloordiv_fullSuite(self):
		""" __rfloordiv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod_rop(self.constructor, numpy.floor_divide, '__rfloordiv__', False, 0)

	def test_rfloordiv_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __rfloordiv__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__rfloordiv__', False)

	def test_rfloordiv_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __rfloordiv__"""
		back_binaryelementwise_name_preservations(self.constructor, '__rfloordiv__', False)


	################
	# __ifloordiv__ #
	################

	def test_ifloordiv_fullSuite(self):
		""" __ifloordiv__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.floor_divide, '__ifloordiv__', True, 0)

	def test_ifloordiv_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __ifloordiv__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__ifloordiv__', True)


	def test_ifloordiv_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __ifloordiv__"""
		back_binaryelementwise_name_preservations(self.constructor, '__ifloordiv__', True)


	###############
	# __mod__ #
	###############

	def test_mod_fullSuite(self):
		""" __mod__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.mod, '__mod__', False, 0)

	def test_mod_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __mod__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__mod__', False)

	def test_mod_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __mod__"""
		back_binaryelementwise_name_preservations(self.constructor, '__mod__', False)


	################
	# __rmod__ #
	################

	def test_rmod_fullSuite(self):
		""" __rmod__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod_rop(self.constructor, numpy.mod, '__rmod__', False, 0)

	def test_rmod_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __rmod__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__rmod__', False)

	def test_rmod_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __rmod__"""
		back_binaryelementwise_name_preservations(self.constructor, '__rmod__', False)


	################
	# __imod__ #
	################

	def test_imod_fullSuite(self):
		""" __imod__ Run the full standardized suite of tests for a binary numeric op """
		run_full_backendDivMod(self.constructor, numpy.mod, '__imod__', True, 0)


	def test_imod_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __imod__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__imod__', True)

	def test_imod_binaryelementwise_name_preservations(self):
		""" Test p/f names are preserved when calling elementwise __imod__"""
		back_binaryelementwise_name_preservations(self.constructor, '__imod__', True)





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

	def test_pow_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __pow__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__pow__', False)


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

	def test_ipow_binaryscalar_name_preservations(self):
		""" Test p/f names are preserved when calling __ipow__ with scalar arg"""
		back_binaryscalar_name_preservations(self.constructor, '__ipow__', True)


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

	def test_pos_unary_name_preservations(self):
		""" Test that point / feature names are preserved when calling __pos__ """
		back_unary_name_preservations(self.constructor, '__pos__')


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

	def test_neg_unary_name_preservations(self):
		""" Test that point / feature names are preserved when calling __neg__ """
		back_unary_name_preservations(self.constructor, '__neg__')


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

	def test_abs_unary_name_preservations(self):
		""" Test that point / feature names are preserved when calling __abs__ """
		back_unary_name_preservations(self.constructor, '__abs__')
