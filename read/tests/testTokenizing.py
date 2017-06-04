"""
Unit tests of the functions to do basic text processing - remove html, tokenize,
remove symbols, remove stop words, etc.
"""

import os
import UML

#from UML.read.text_processing import readDirectoryIntoSparseMatrix
from UML.read.text_processing import convertToTokens
#from UML.read.text_processing import convertAttributeMapToMatrix

testDirectory = os.path.join(UML.UMLPath, 'read', 'tests', 'testDirectory')


def test_defaultConvertToTokens():
    """
    Unit test of the convertToTokens function in uml_loader
    """
    testText1 = "<html><body><p>Hello how are you?</p><p>It's a great day!?!?!</p>"
    testText2 = """Hello, how are you, John? When the roses are in bloom,
	dogs can't run willy-nilly. 
	It's a great day! My email is dog@cat.com.  Fish&Chips."""
    testTokens1, testFreqMap1 = convertToTokens(testText1)
    testTokens2, testFreqMap2 = convertToTokens(testText2)

    #assertions related to testText1
    assert len(testTokens1) == 3
    assert 'hello' in testTokens1
    assert 'gre' in testTokens1
    assert 'day' in testTokens1
    assert testFreqMap1['hello'] == 1
    assert testFreqMap1['gre'] == 1
    assert testFreqMap1['day'] == 1

    #assertions related to testText2
    assert len(testTokens2) == 13
    assert 'hello' in testTokens2
    assert 'john' in testTokens2
    assert 'ros' in testTokens2
    assert 'bloom' in testTokens2
    assert 'willynil' in testTokens2
    assert 'run' in testTokens2
    assert 'gre' in testTokens2
    assert 'day' in testTokens2
    assert testFreqMap2['hello'] == 1
    assert testFreqMap2['john'] == 1
    assert testFreqMap2['ros'] == 1
    assert testFreqMap2['willynil'] == 1
    assert testFreqMap2['gre'] == 1
    assert testFreqMap2['day'] == 1


def test_removeContainingConvertToTokens():
    """
    Unit test of removeTokensContaining feature of convertToTokens function
    """
    testText1 = """H#llo, h&w a%e y!u, J+hn? W=en t>e r*ses are in bl^om,
	dogs can't run willy-nilly. 
	It's a great day! My email is dog@cat.com"""
    testTokens1, testFreqMap1 = convertToTokens(testText1,
                                                removeTokensContaining=['@', '#', '+', '%', '&', '=', '^', '>', '!'])
    assert 'h#llo' not in testTokens1
    assert 'hllo' not in testTokens1
    assert 'h&w' not in testTokens1
    assert 'j+hn' not in testTokens1
    assert 'email' in testTokens1


def test_keepNumbers():
    """
    Unit test of removing/not removing numbers (digits) from tokenized text.
    """
    testText = """Hello, how are 1st you, John? When 234 the roses are in bloom,
				dogs can't run 2nd willy-nilly. 5s 234har908247134
				It's a grea7890t day! My email is dog@cat.com.  Fish&Chips."""

    testTokens1, testTokenFreqMap1 = convertToTokens(testText, keepNumbers=True)
    testTokens2, testTokenFreqMap2 = convertToTokens(testText, keepNumbers=False)

    from UML.read.defaults import numericalChars

    for token in testTokens2:
        tokenSet = set(token)
        assert tokenSet.isdisjoint(numericalChars)

    assert '234' in testTokens1
    assert '234har908247134' in testTokens1
    assert 'grea7890t' in testTokens1



