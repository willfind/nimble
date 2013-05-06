from nltk.corpus import stopwords

defaultSkipSetNonAlpha = ''.join(c for c in map(chr, range(256)) if not c.isalpha())
defaultSkipSetNonAlphaNumeric = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
numericalChars = ''.join(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
defaultStopWords = stopwords.words('english')
