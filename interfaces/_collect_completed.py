
import os
import importlib
import abc
import universal_interface

def collect(modulePath):
	# go through files in this directory, find ones which could be python importable
	possibleFiles = os.listdir(modulePath)
	pythonModules = []
	for fileName in possibleFiles:
		if '.' not in fileName:
			continue
		(name, extension) = fileName.split('.')
		if extension == 'py' and not name.startswith('_'):
			pythonModules.append(name)

	# go through each possible module, import it, and check for possible interfaces
	possibleInterfaces = []
	# setup seen with the interfaces we know we don't want to load / try to load
	seen = set(["UniversalInterface", "UniversalInterfaceLookalike", "CustomLearnerInterface"])
	for toImport in pythonModules:
		importedModule = importlib.import_module('.' + toImport, __package__)
		contents = dir(importedModule)

		# for each attribute of the module, we will check to see if it is a subclass of
		# the UniversalInterface
		for valueName in contents:
			value = getattr(importedModule, valueName)
			if isinstance(value, abc.ABCMeta) and issubclass(value, universal_interface.UniversalInterface):
				if not value in seen:
					seen.add(value)
					possibleInterfaces.append(value)				

	# We now have a list of possible interfaces, which we will try to instantiate
	instantiated = []
	for toInstantiate in possibleInterfaces:
		tempObj = None
		try:
			tempObj = toInstantiate()
		# if ANYTHING goes wrong, just go on without that interface
		except Exception:
			continue
		if tempObj is not None:
			instantiated.append(tempObj)
	# key: canonical names
	# value: interface using that name, or None if that name has a collision
	nameToInterface = {}
	for namedInterface in instantiated:
		canonicalName = namedInterface.getCanonicalName()
		if canonicalName in nameToInterface.keys():
				# TODO register error with subsystem
				nameToInterface[canonicalName] = None
		else:
			nameToInterface[canonicalName] = namedInterface

	# interfaces should not accept as aliases the canonical names of other interfaces
	for currName in nameToInterface:
		if nameToInterface[currName] is None:
			continue
		for checkName in nameToInterface:
			# we only care about valid interfaces other than currName's
			if currName != checkName and nameToInterface[checkName] is not None:
				# if currName's interface accepts another's canonical name, we erase it
				if nameToInterface[currName].isAlias(checkName):
					# TODO register error with subsystem
					nameToInterface[currName] = None
					break

	validatedByName = []
	for name in nameToInterface:
		if nameToInterface[name] is not None:
			validatedByName.append(nameToInterface[name])

	return validatedByName
