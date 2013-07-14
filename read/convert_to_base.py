from UML.read.defaults import defaultSkipSetNonAlphaNumeric
from UML.read.defaults import defaultStopWords
from dok_data_set import DokDataSet
from ClassLabelMap import ClassLabelMap
from UML.exceptions import ArgumentException


def convertToCooBaseData(dirPath=None, fileExtensions=['.txt', '.html'], 
                         dirMappingMode='all', 
                         attributeMaps=None,
                         attributeTransformFunctionsMap=None,
                         docIdClassLabelMaps=None,
                         requiredClassLabelTypes=None,
                         minTermFrequency=2,
                         featureTypeWeightScheme=None,
                         featureRepresentation='frequency',
                         cleanHtml=True,
                         ignoreCase=True,
                         tokenizer='default',
                         removeBlankTokens=True,
                         skipSymbolSet=defaultSkipSetNonAlphaNumeric,
                         removeTokensContaining=None,
                         keepNumbers=False, stopWordSet=defaultStopWords,
                         tokenTransformFunction=None,
                         stemmer='default'):
    """
    Umbrella function to manage process of converting raw data (in the form of a directory of individual files
    on the file system, or a mapping between document ID and attribute) into a Coo BaseData object, which can
    be used as any other Coo BaseData object or written to disk for use later on.  

    For files on disk, this function assumes that they contain raw text, which must be tokenized and processed
    in order to convert each document to a form which can be passed as input to an ML algorithm.  The details of
    text pre-processing can be controlled through parameters of this function.

    The organization of the final BaseData object will have one document per row.  The leftmost column in the
    object should contain document ID's; then one or more columns containing class labels, then columns
    representing features.

    Parameters:
        dirPath: path to a directory on the filesystem which contains text files to be loaded and processed.
                 All files associated with the same document must have the same filename; filenames should be
                 numbers.  The final BaseData object will have a column named docId*** which will contain
                 document ID numbers.  Text in all files with the same name/ID will in effect be concatenated,
                 depending on the value of dirMappingMode.

        fileExtensions: List of strings representing file extensions, in the form '.XXX'.  Files with these 
                        extensions will be loaded and added to the final object; all other files will be ignored.
                        If the list is blank, all files will be loaded (not recommended).

        dirMappingMode: dirMappingMode has two possible values: 'all' and 'multiTyped'..  In 'all' mode, every 
                        single file in the directory pointed to by dirPath, and all of its subdirectories, will be loaded 
                        (as long as it has a desired extension), and all tokens will be treated as coming 
                        from the same pool.  So if the same token occurs in three files with the same filename,
                        it will always be treated as the same token. 'multiTyped' mode allows for separating files
                        (and thus tokens) based on their type/source, to allow a document to consist of multiple 
                        different types of text.  In 'multiTyped' mode, only files which reside within one of dirPath's 
                        immediate subdirectories (or any of their subdirectories, recursive) will be loaded.  If 
                        dirPath contains two subdirectories named 'head' and 'body,' then all files in head 
                        will be loaded and processed separately from all files in body.  Though a file in head
                        with the name 001.txt will eventually be combined with a file named 001.txt in body
                        into one document, tokens from 'head' will be prepended with the name 'head_', so that
                        they are never the same feature as tokens from files within the 'body' subdirectory.

                        Note:  in 'multiTyped' mode, files whose parent directory is dirPath, and not a subdirectory
                        of dirPath, will be ignored, as the 'type' of a file depends on the name of the 
                        child directory of dirPath within which it resides.

        attributeMaps: A map of maps, of the form {attributeName:{docId:attributeContent}}.  A data set is thus
                       allowed to have multiple attribute types.  Each document is allowed to have zero or one
                       attributes from each attribute type.  Attributes may be strings, but it is assumed that
                       they do not require the same amount of processing as raw text data.  If it is necessary
                       to do some processing on attributes before they are converted to features, this processing
                       must be contained within a function and added to attributeTransformFunctionsMap. Each
                       unique attribute from each attributeType will be represented as a feature column in the final
                       BaseData object.  

        attributeTransformFunctionsMap: A map of functions, of the form attributeName:attributeTransformFunction.
                       attributeNames must be the same as those used in the attributeMaps parameter.  If there
                       is an entry for a given attributeName, it will be applied to all entries in the associated
                       map in attributeMaps.

        docIdClassLabelMaps: Map of maps of the form {classLabelType:{docId:classLabel}}.  Allows for a data set
                             to contain multiple types of class label.  Each classLabelType will be represented
                             by one column in the final object; the name of the column will be classLabelType -
                             the key of the mapping of docId:classLabel for those values.  

                             TODO: allow for a case where there is only one set of class labels, use default
                             classLabel name.

        requiredClassLabelTypes:  List of classLabelTypes (must be the same as those used in docIdClassLabelMaps)
                            that each document must possess in order to be retained in the returned data set.
                            For example: if there are three classLabelTypes A, B, and C; and A and B are in 
                            requiredClassLabelTypes, then any document that doesn't have an A label *and* a B
                            label will not be included in the results.  

        minTermFrequency: Minimun number of documents a feature has to appear in to be included in the final
                          data set.  Any feature that does not appear in at least this many documents will be
                          removed from the data set.

        featureRepresentation: Three possible values: 'binary', 'frequency', 'tfidf'.  If 'binary' is chosen,
                               the final object will have an entry of 1 for each feature present in each document.
                               If 'frequency' is chosen, the # of occurences of each feature in each document
                               will be recorded. If 'tfidf' is chose, the TF-IDF value for each feature in each
                               document is recorded.

        ***********************************************************************************************************

        The following parameters relate to the process of tokenizing/pre-processing of text.  They are the
        same as the parameters of the convertToTokens() function in text_processing.py.  For a more in-depth
        discussion of the parameters and implementation of tokenizing/processing, see docstring for 
        convertToTokens().

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

        removeSymbolsContaining: A set of symbols.  Any token containing any symbol/character in this set will be
                                 removed entirely.  If a symbol appears in both this set and in skipSymbolSet,
                                 removeSymbolsContaining will take precedence, and all tokes containing that symbol
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
    if dirPath is None and (attributeMaps is None or len(attributeMaps) == 0):
        raise ArgumentException("Either dirPath or attributeMaps must be present")

    #We need to create textDataSet, though we may not use it
    textDataSet = None
    #create new DokDataSet object; DokDataSet class does most of the work
    if dirPath is not None:
        textDataSet = DokDataSet()
        textDataSet.loadDirectory(dirPath, fileExtensions, dirMappingMode, cleanHtml, ignoreCase, tokenizer, removeBlankTokens, skipSymbolSet, removeTokensContaining, keepNumbers, stopWordSet, tokenTransformFunction, stemmer)

    #load all attribute sets as separate DokDataSet objects
    attrDataSets = []
    if attributeMaps is not None:
        for attributeName, docIdAttrMap in attributeMaps.iteritems():
            if attributeTransformFunctionsMap is not None and attributeName in attributeTransformFunctionsMap:
                attributeTransformFunction = attributeTransformFunctionsMap[attributeName]
            else:
                attributeTransformFunction = None
            attrDataSet = DokDataSet()
            attrDataSet.loadAttributeMap(docIdAttrMap, attributeName, attributeTransformFunction)
            attrDataSets.append(attrDataSet)

    #merge all the DokDataSet objects together.  Merge is essentially a left-join, with the
    #dokDataSet object that calls merge() on the left.
    dataSet = None
    if textDataSet is not None:
        if len(attrDataSets) > 0:
            for attrDataSet in attrDataSets:
                textDataSet.merge(attrDataSet)
        dataSet = textDataSet
    else:
        if len(attrDataSets) > 1:
            firstAttrDataSet = attrDataSets[0]
            i = 1
            while(i < len(attrDataSets)):
                attrDataSet = attrDataSets[i]
                firstAttrDataSet.merge(attrDataSet)
                i += 1
            dataSet = firstAttrDataSet
        else:
            dataSet = attrDataSets[0]

    #Add all classLabel sets to main dataSet object, checking to see if they are required or not.
    if docIdClassLabelMaps is not None:
        for classLabelName, docIdClassLabelMap in docIdClassLabelMaps.iteritems():
            if requiredClassLabelTypes is not None and classLabelName in requiredClassLabelTypes:
                isRequired = True
            else:
                isRequired = False
            classLabelMap = ClassLabelMap(docIdClassLabelMap, classLabelName, isRequired)
            dataSet.addClassLabelMap(classLabelMap)

    return dataSet.toCooBaseData(featureRepresentation, minTermFrequency, featureTypeWeightScheme)
