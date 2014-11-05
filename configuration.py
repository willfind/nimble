"""
Contains code relating to file I/O and manipulation of the optional
configuration of UML.

During UML initialization, there is a specific order to tasks relating
to configuration. Before any operation that might rely on being able to
access UML.settings (for example, interface initialization) we must load
it from file, so that in the normal course of operations, user set values
are available to be used accross UML. Alternatively, in the case that there
is a new source of user set options (for example, an interface that has
been loaded for the first time) we still load from the config file first,
then do initialization of all of the those modules that might need access
to UML.settings, using hard coded defaults if needed, then at the end
of UML initialization we will always perform a syncing helper, which
will ensure that the configuration file reflects all available options.

"""

# Note: .ini format's option names are not case sensitive?


import ConfigParser
import os
import copy

import UML
from UML.exceptions import ArgumentException

class SortedCommentPreservingConfigParser(ConfigParser.SafeConfigParser):

	def _getComment(self, section, option):
		"""
		Returns None if the appropriate section doesn't exist
		"""
		try:
			return self._comments[section][option]
		except:
			return None


	def write(self, fp):
		"""Write an .ini-format representation of the configuration state,
		including any saved comments from when loaded.
		"""
		if self._defaults:
			secComment = self._getComment(ConfigParser.DEFAULTSECT, None)
			if secComment is not None:
				fp.write(secComment)

			fp.write("[%s]\n" % ConfigParser.DEFAULTSECT)
			for (key, value) in self._defaults.items():
				optComment = self._getComment(ConfigParser.DEFAULTSECT, key)
				if optComment is not None:
					fp.write(optComment)
				fp.write("%s = %s\n" % (key, str(value).replace('\n', '\n\t')))
			fp.write("\n")
		sortedSections = sorted(self._sections)
		for section in sortedSections:
			secComment = self._getComment(section, None)
			if secComment is not None:
				fp.write(secComment)

			fp.write("[%s]\n" % section)
			sortedKV = sorted(self._sections[section].items())
			for (key, value) in sortedKV:
				if key == "__name__":
					continue

				optComment = self._getComment(section, key)
				if optComment is not None:
					fp.write(optComment)

				if (value is not None) or (self._optcre == self.OPTCRE):
					key = " = ".join((key, str(value).replace('\n', '\n\t')))
				fp.write("%s\n" % (key))
			fp.write("\n")

	def _read(self, fp, fpname):
		"""Parse a sectioned setup file.

		The sections in setup file contains a title line at the top,
		indicated by a name in square brackets (`[]'), plus key/value
		options lines, indicated by `name: value' format lines.
		Continuations are represented by an embedded newline then
		leading whitespace.  Blank lines, and just about everything
		else is ignored, excepting comments.

		Comment lines are saved, associated with the next entry below
		them, so they can be written back during write().
		"""
		cursect = None                        # None, or a dictionary
		optname = None
		lineno = 0
		e = None                              # None, or an exception
		currComment = None
		self._comments = self._dict()
		comments = self._comments
		while True:
			line = fp.readline()
			if not line:
				break
			lineno = lineno + 1
			# comment or blank line?
			if line.strip() == '':
				continue
			# save any comments, then continue
			if line.strip() == '' or line[0] in '#;':
				if currComment is None:
					currComment = line
				else:
					currComment += line
				
				continue

			if line.split(None, 1)[0].lower() == 'rem' and line[0] in "rR":
				# no leading whitespace
				continue
			# continuation line?
			if line[0].isspace() and cursect is not None and optname:
				value = line.strip()
				if value:
					cursect[optname].append(value)
			# a section header or option header?
			else:
				# is it a section header?
				mo = self.SECTCRE.match(line)
				if mo:
					sectname = mo.group('header')
					if sectname in self._sections:
						cursect = self._sections[sectname]
					elif sectname == ConfigParser.DEFAULTSECT:
						cursect = self._defaults

						comments[sectname] = self._dict()
						comments[sectname][None] = currComment
						currComment = None
					else:
						cursect = self._dict()
						cursect['__name__'] = sectname
						self._sections[sectname] = cursect

						comments[sectname] = self._dict()
						comments[sectname][None] = currComment
						currComment = None
					# So sections can't start with a continuation line
					optname = None
				# no section header in the file?
				elif cursect is None:
					raise ConfigParser.MissingSectionHeaderError(fpname, lineno, line)
				# an option line?
				else:
					mo = self._optcre.match(line)
					if mo:
						optname, vi, optval = mo.group('option', 'vi', 'value')
						optname = self.optionxform(optname.rstrip())
						# This check is fine because the OPTCRE cannot
						# match if it would set optval to None
						if optval is not None:
							if vi in ('=', ':') and ';' in optval:
								# ';' is a comment delimiter only if it follows
								# a spacing character
								pos = optval.find(';')
								if pos != -1 and optval[pos-1].isspace():
									optval = optval[:pos]
							optval = optval.strip()
							# allow empty values
							if optval == '""':
								optval = ''
							cursect[optname] = [optval]
						else:
							# valueless option handling
							cursect[optname] = optval

						comments[sectname][optname] = currComment
						currComment = None
					else:
						# a non-fatal parsing error occurred.  set up the
						# exception but keep going. the exception will be
						# raised at the end of the file and will contain a
						# list of all bogus lines
						if not e:
							e = ConfigParser.ParsingError(fpname)
						e.append(lineno, repr(line))
		# if any parsing errors occurred, raise an exception
		if e:
			raise e

		# join the multi-line values collected while reading
		all_sections = [self._defaults]
		all_sections.extend(self._sections.values())
		for options in all_sections:
			for name, val in options.items():
				if isinstance(val, list):
					options[name] = '\n'.join(val)



class SessionConfiguration(object):
	"""
	Class through which UML user interacts with the saveable configuration
	options define behavior dependent on the host system. The backend is
	a SortedConfigParser (essentially a SafeConfigParser where sections and
	options are written in sorted order) object which deals with the file
	I/O of the INI formated file on disk. This wrapper class allows for
	temporary changes to that configuration set and gives the user control
	over which changes should be saved and when.

	"""

	def __init__(self, path):
		self.cp = SortedCommentPreservingConfigParser()
		# Needs to be set if you want option names to be case sensitive
		self.cp.optionxform = str
		self.cp.read(path)
		self.path = path

		self.changes = {}

	def delete(self, section, option):
		success = False

		if section is not None:
			# Deleting a specific option in a specific section
			if option is not None:		
				if (section, option) in self.changes:
					del self.changes[(section,option)]
					success = True
				if self.cp.has_section(section):
					self.cp.remove_option(section, option)
			# deleting an entire section
			else:
				for (k,v) in self.changes:
					if k == section:
						del self.changes[(k,v)]
						success = True
				if self.cp.has_section(section):
					self.cp.remove_section(section)
		else:
			if option is not None:
				msg = "If specifying and option, one must also specify "
				msg += "a section"
				raise ArgumentException(msg)
			else:
				pass  # if None, None is specified, we will return false
		return success


	def get(self, section, option):
		# Treat this as a request for an entire section
		if section is not None and option is None:
			found = False
			ret = {}
			if self.cp.has_section(section):
				found = True
				for (k,v) in self.cp.items(section):
					ret[k] = v
			for (kSec,kOpt) in self.changes:
				if kSec == section:
					found = True
					ret[kOpt] = self.changes[(kSec,kOpt)]
			if not found:
				raise ConfigParser.NoSectionError()
			return ret
		# Otherwise, treat it as a request for a single option,
		# and let the helpers deal with the failure modes
		else:
			if (section, option) in self.changes:
				return self.changes[(section, option)]
			fromFile = self.cp.get(section, option)
			return fromFile

	def set(self, section, option, value):
		"""
		Set an option for this session.
		"""
		# if we are setting a value which matches the
		# value in file, we should remove the entry
		# from the changes dict
		try:
			inFile = self.cp.get(section, option)
			if inFile == value:
				if (section, option) in self.changes:
					del self.changes[(section, option)]
					return
		except:  # TODO make this specific
			pass

		# check: is this section the name of an interface
		try:
			ignore = True
			# raises argument exception if not an interface name
			interface = UML.umlHelpers.findBestInterface(section)
			ignore = False
			acceptedNames = interface.optionNames
			if option not in acceptedNames:
				msg = section + " is associated with an interface "
				msg += "which only allows options: " 
				msg += str(acceptedNames)
				msg += " but " + option + " was given"
				raise ArgumentException(msg)
		except ArgumentException as e:
			if not ignore:
				raise e
		self.changes[(section, option)] = value

	def saveChanges(self, section=None, option=None):
		"""
		Depending on the values of the inputs, three levels of saving
		are allowed. If both section and option are specified, that
		specific value will be written to file. If only section is
		specified and option is None, then any changes in that section
		will be written to file. And if both section and option are None,
		then all changes will be written to file. If section is None
		and option is not None, an exception is raised.

		"""
		# if no changes have been made, nothing needs to be saved.
		if self.changes == {}:
			return

		if section is None:
			if option is not None:
				raise UML.exceptions.ArgumentException("If section is None, option must also be None")
			# save all
			for (sec, opt) in self.changes.keys():
				if not self.cp.has_section(sec):
					self.cp.add_section(sec)
				self.cp.set(sec, opt, self.changes[(sec,opt)])
			self.changes = {}
		else:
			if option is None:
				#save section
				for (sec, opt) in self.changes.keys():
					if sec != section:
						continue
					if not self.cp.has_section(sec):
						self.cp.add_section(sec)
					self.cp.set(sec, opt, self.changes[(sec,opt)])
					del self.changes[(sec,opt)]
			else:
				# save specific
				for (sec, opt) in self.changes.keys():
					if sec != section or opt != option:
						continue
					if not self.cp.has_section(sec):
						self.cp.add_section(sec)
					self.cp.set(sec, opt, self.changes[(sec,opt)])
					del self.changes[(sec,opt)]

		fp = open(self.path, 'w')
		self.cp.write(fp)
		fp.close()


def loadSettings():
	target = os.path.join(UML.UMLPath, 'configuration.ini')

	if not os.path.exists(target):
		fp = open(target, 'w')
		fp.close()

	ret = SessionConfiguration(target)
	return ret

def syncWithInterfaces(settingsObj):
	origChanges = copy.copy(settingsObj.changes)
	newChanges = {}
	settingsObj.changes == {}

	# iterate through interfaces?
	for interface in UML.interfaces.available:
		interfaceName = interface.getCanonicalName()
		optionNames = interface.optionNames

		# check that all present are valid names
		if settingsObj.cp.has_section(interfaceName):
			for (opName, value) in settingsObj.cp.items(interfaceName):
				if opName not in optionNames:
					settingsObj.cp.remove_option(interfaceName, opName)
		else:
			# We only want to add sections for those interfaces which
			# actually have options
			if len(optionNames) > 0:
				settingsObj.cp.add_section(interfaceName)

		# check that all names are present, add if not
		for opName in optionNames:
			try:
				settingsObj.get(interfaceName, opName)
			except ConfigParser.NoOptionError:
				settingsObj.set(interfaceName, opName, "")
			if (interfaceName, opName) in origChanges:
				newChanges[(interfaceName, opName)] = origChanges[(interfaceName, opName)]

	# save all changes that were made
	settingsObj.saveChanges()

	# NOTE: this is after the save. If a non-empty change set was in place
	# during the save operation, those changes would be reflected in
	# the file, which we don't want. We instead want to preserve the
	# user changes, and only save them because of user intervention.
	settingsObj.changes = newChanges
