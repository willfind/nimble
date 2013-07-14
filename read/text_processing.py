import nltk
import types
import math
from collections import Counter
from scipy import *
from scipy.sparse import *
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.stem.lancaster import LancasterStemmer

from UML.read.defaults import defaultSkipSetNonAlphaNumeric
from UML.read.defaults import defaultStopWords
from UML.read.defaults import numericalChars
from UML.exceptions import ArgumentException, EmptyFileException


def loadAndTokenize(textFilePath, 
				    cleanHtml=True, 
				    ignoreCase=True, 
				    tokenizer='default', 
				    removeBlankTokens=True, 
				    skipSymbolSet=defaultSkipSetNonAlphaNumeric, 
				    removeTokensContaining=None, 
				    keepNumbers=False, 
				    stopWordSet=defaultStopWords, 
				    tokenTransformFunction=None, 
				    stemmer='default'):
	"""
	Load one file, run the contents of that file through text-processing function (convertToTokens), and return
	the contents as a list of tokens and a frequency map in the form token:# of occurences in file.
	"""
	if not isinstance(textFilePath, str):
		raise ArgumentException("Invalid argument to defaultLoadAndTokenize: "+repr(textFilePath))

	textFile = open(textFilePath, 'rU')

	text = textFile.read()

	if text is None or text == '':
		raise EmptyFileException("Attempted to tokenize empty file in loadAndTokenize")

	tokens, frequencyMap = convertToTokens(text, cleanHtml, ignoreCase, tokenizer, removeBlankTokens, skipSymbolSet, removeTokensContaining, keepNumbers, stopWordSet, tokenTransformFunction, stemmer)

	textFile.close()
	
	return tokens, frequencyMap


def convertToTokens(text, 
					cleanHtml=True, 
					ignoreCase=True, 
					tokenizer='default', 
					removeBlankTokens=True, 
					skipSymbolSet=defaultSkipSetNonAlphaNumeric, 
					removeTokensContaining=None, 
					keepNumbers=False, 
					stopWordSet=defaultStopWords, 
					tokenTransformFunction=None, 
					stemmer='default'):
	"""
		Convert raw text to processed tokens.  Returns a list of tokens in the same order in
		which they appear in the original text, as well as a map of token:frequencyCount.  
		The only required input is the text to be processed.  If the user wants to use their
		own versions of any of the various default processing tasks, they can be passed as
		arguments.  By default, removes html tags, does tokenizing, symbol filtering, removal of stop words,
		removal of numeric characters, and stemming.

		Inputs:

		text - raw text to be tokenized, as a string

		cleanHtml: boolean.  Set to true if expect text to contain html tags that should be removed.  If text
                   does not contain html, attempting to remove html will not likely result in any
                   changes, so defaults to true.

        ignoreCase:  boolean.  If True, all text is converted to lowercase in the course of pre-processing.  If
                     False, tokens with the same spelling but different capitalization will be considered
                     different tokens

        tokenizer:  Designator of which tokenizing function to use.  If None, use the default tokenizer, which
                    is the Punkt tokenizer in the nltk package.  If tokenizer is a string, we assume that it is
                    the name of a tokenizer in the nltk package, and attempt to import it.  Otherwise, tokenizer
                    must be a function.

        removeBlankTokens: boolean flag.  If True, will remove all blank tokens present after tokenizing.  If False,
                           blank tokens present after tokenizing will not be removed.  Blank tokens created by
                           symbol removal will still be removed, even if removeBlankTokens is False, to prevent
                           confusion between blank tokens created by tokenizing and those created by removing symbols.

        skipSymbolSet: A set of symbols.  These symbols will be removed from tokens, leaving the rest of the token
                       intact.  Defaults to all non alphanumeric characters.

        removeTokensContaining: A set of symbols.  Any token containing any symbol/character in this set will be
                                 removed entirely.  If a symbol appears in both this set and in skipSymbolSet,
                                 removeTokensContaining will take precedence, and all tokens containing that symbol
                                 will be removed.  Defaults to None.

        keepNumbers: boolean flag.  If true, numerals will not be removed.  If False, numerals will be treated
                     as symbols in skipSybolSet:  they will be removed from tokens, but a token containing numeric
                     and non-numeric symbols will have the numerals removed, but not the non-numerals.  Thus,
                     '2nd' would become 'nd'.  If 'nd' should also be removed, it should be added to the stop list.

        stopWordSet: Set of words (tokens) to be removed from the final set of tokens.  Defaults to nltk's default
                     english-language stop word list.

        tokenTransformFunction:  If the user wishes to add an additional function to the tokenizing/pre-processing
                                 process, it can be passed as the tokenTransformFunction.  This function will be
                                 called on each token just before stemming, which is the last step in the process.

        stemmer: Stemmer function, or 'default'.  If 'default,' nltk's LancasterStemmer is used to stem tokens.
                 Otherwise, use the passed function.
	"""
	if text is None or text == '':
		raise ArgumentException("convertToTokens requires a non-empty input string")

	if cleanHtml:
		text = nltk.clean_html(text)

	if tokenizer == 'default':
		tokenizer = PunktWordTokenizer()
		tokens = tokenizer.tokenize(text)
	elif isinstance(tokenizer, types.FunctionType):
		tokens = tokenizer(text)
	elif isinstance(tokenizer, str):
		#If tokenizer is a string other than 'default', we assume that it's the name
		# of a tokenizer in nltk's tokenize package
		if tokenizer.startswith("nltk.tokenize"):
			moduleName = tokenizer.rpartition('.')[0]
		else:
			moduleName = "nltk.tokenize"+tokenizer.rpartition('.')[0]
		className = tokenizer.rpartition('.')[2]
		exec("from "+moduleName+" import "+className)
		exec("tokenizerObj = "+className+"()")
		tokens = tokenizerObj.tokenize(text)
	else:
		raise ArgumentException("convertToTokens requires a tokenizer")

	if ignoreCase:
		lowerCaseTokens = []
		for token in tokens:
			lowerCaseTokens.append(token.lower())
		tokens = lowerCaseTokens

	#set the collection of stopwords to be used, if stop word filtering is turned on
	if stopWordSet is not None:
		#Set the set of stop words: read in a file, use a passed list, or use the default list
		if isinstance(stopWordSet, str):
			stopWordSet = loadStopList(stopWordSet)
		elif isinstance(stopWordSet, (list, set)):
			pass
		else:
			raise ArgumentException("Unrecognized stop list type in convertToTokens: "+type(stopWordSet))

		#filter out tokens that are in the set of stop words
		tokens = filterStopWords(tokens, stopWordSet)

	#Remove all tokens containing any symbol in the set removeTokensContaining
	if removeTokensContaining is not None:
		flagSymbolFilteredSet = []
		for token in tokens:
			removeToken = False
			for character in token:
				if character in removeTokensContaining:
					removeToken = True
			if not removeToken:
				flagSymbolFilteredSet.append(token)
		tokens = flagSymbolFilteredSet

	if skipSymbolSet is not None:
		filteredSet = []
		for token in tokens:
			if token == '':
				if not removeBlankTokens:
					filteredSet.append(token)
			else:
				filteredToken = token.translate(None, skipSymbolSet)
				if filteredToken != '':
					filteredSet.append(filteredToken)
		tokens = filteredSet

	if not keepNumbers:
		filteredSet = []
		numbersSet = numericalChars
		for token in tokens:
			if token == '' and not removeBlankTokens:
				filteredSet.append(token)
			else:
				filteredToken = token.translate(None, numbersSet)
				if filteredToken != '':
					filteredSet.append(filteredToken)
		tokens = filteredSet

	# May need to run stop word filtering again, as there may be 
	# more matches after symbol removal
	if stopWordSet is not None and skipSymbolSet is not None:
		tokens = filterStopWords(tokens, stopWordSet)

	if tokenTransformFunction is not None:
		transformedSet = []
		for token in tokens:
			transformedSet.append(tokenTransformFunction(token))
		tokens = transformedSet

	if stemmer == 'default':
		stemmer = LancasterStemmer()
		stemmedTokens = []
		for token in tokens:
			stemmedTokens.append(stemmer.stem(token))
		tokens = stemmedTokens
	else:
		stemmedTokens = []
		for token in tokens:
			stemmedTokens.append(stemmer(token))
		tokens = stemmedTokens

	frequencyMap = convertTokenListToFrequencyMap(tokens)

	return tokens, frequencyMap


def filterStopWords(tokens, stopWords=set()):
	"""
	Provided a list of tokens and a set of stop words, return a list of tokens
	from which stop words have been removed. Original order is maintained, 
	minus any tokens that have been removed.
	"""
	if tokens is None:
		raise ArgumentException("List of tokens in filterStopWords must not be None")

	filteredList = []
	for token in tokens:
		if token not in stopWords:
			filteredList.append(token)

	return filteredList


def convertTokenListToFrequencyMap(tokenList):
	"""
	Given a string containing text, convert to a map of 
	token:tokenFrequency, as a collections.Counter object
	"""
	if tokenList is None or len(tokenList) == 0:
		return None

	#Convert text to tokens using convertTextToTokenList, possibly using
	#user-defined function
	tokenCountMap = Counter()
	for token in tokenList:
		if token in tokenCountMap:
			tokenCountMap[token] = tokenCountMap[token] + 1
		else:
			tokenCountMap[token] = 1

	return tokenCountMap

def computeCorpusTermCountMap(docTermCountsList):
	"""
	Provided a list of dictionaries, where each dictionary maps tokens
	to frequency counts and represents one document, return a dictionary
	associating each token in the corpus with the number of documents in which
	that token appears.
	"""
	corpusDocumentTermFrequencies = {}
	for doc in docTermCountsList:
		for term, count in doc.iteritems():
			if term in corpusDocumentTermFrequencies and count > 0:
				corpusDocumentTermFrequencies[term] += 1
			else:
				corpusDocumentTermFrequencies[term] = 1

	return corpusDocumentTermFrequencies

def loadStopList(stopListFilePath=None, ignoreCase=True):
	"""
	Load a stop list from the specified file.  If no file is specified, load
	the default nltk stop word list.  If a filePath is passed, assumes that
	the file has one stop word per line.
	"""
	if stopListFilePath is None or not isinstance(stopListFilePath, str):
		raise ArgumentException("stopListFilePath must be a string pointing to the location of " + 
								"a file containing stop words")

	stopListFile = open(stopListFilePath, 'rU')
	stopList = []
	for line in stopListFile:
		if line != '':
			if ignoreCase:
				stopList.append(line.lower().strip())
			else:
				stopList.append(line.strip())

	return stopList

