"""
Functions for creating a nice string representation of the tables
generated in .data_set_analyzer.
"""

import copy
import math


class TableError(Exception):
    """
    Raise when a table cannot be created.
    """


def objectClass(obj):
    """
    Returns a string representing the class of the given object.
    """
    try:
        #return str(type(obj)).split("'")[1]
        splitClassString = str(obj.__class__).split(".")[1]
        return splitClassString.replace("<type '", "").replace("'>", "")
    except IndexError:
        return str(type(obj)).replace("<type '", "").replace("'>", "")


def tableString(table, rowHeader=True, headers=None, roundDigits=None,
                columnSeparator="", maxRowsToShow=None, snipIndex=None,
                useSpaces=True, includeTrailingNewLine=True):
    """
    Take a table (rows and columns of strings) and return a string
    representing a nice visual representation of that table. roundDigits
    is the number of digits to round floats to
    """
    if not isinstance(table, (list, tuple)):
        raise TableError("table must be list or tuple")
    if len(table) == 0:
        return ""
    if not isinstance(table[0], (list, tuple)):
        msg = "table elements be lists or tuples. You gave: "
        msg += str(objectClass(table[0]))
        raise TableError(msg)

    if isinstance(roundDigits, int):
        roundDigits = "." + str(roundDigits) + "f"

    # So that the table doesn't get destroyed in the process!
    table = copy.deepcopy(table)
    colWidths = []
    #rows = len(table)
    cols = 0
    for row in table:
        if not isinstance(row, list):
            msg = "table must be a list of lists but found a row that had "
            msg += " the value " + str(row)
            raise TableError(msg)
        if len(row) > cols:
            cols = len(row)

    for _ in range(cols):
        colWidths.append(1)

    for i, row in enumerate(table):
        for j, val in enumerate(row):
            if roundDigits is not None and isinstance(val, float):
                toString = format(val, roundDigits)
            else:
                toString = str(val)
            table[i][j] = toString
            if len(toString) > colWidths[j]:
                colWidths[j] = len(toString)

    if headers is not None:
        if len(headers) != cols:
            msg = "Number of table columns (" + str(cols)
            msg += ")  does not match number of header columns ("
            msg += str(len(headers)) + ")!"
            raise TableError(msg)
        for j, header in enumerate(headers):
            if colWidths[j] < len(header):
                colWidths[j] = len(header)

    # if there is a limit to how many rows we can show, delete the middle rows
    # and replace them with a "..." row
    if maxRowsToShow is not None:
        numToDelete = max(len(table) - maxRowsToShow, 0)
        if numToDelete > 0:
            firstToDelete = int(math.ceil((len(table) / 2.0)
                                          - (numToDelete / 2.0)))
            lastToDelete = firstToDelete + numToDelete - 1
            table = table[:firstToDelete]
            table += [["..."] * len(table[firstToDelete])]
            table += table[lastToDelete + 1:]
    # if we want to imply the existence of more rows, but they're not currently
    # present in the table, we just add an elipses at the specified index
    elif snipIndex is not None and snipIndex > 0:
        table = table[:snipIndex]
        table += [["..."] * len(table[0])]
        table += table[snipIndex + 1:]

    #modify the text in each column to give it the right length
    for i, row in enumerate(table):
        for j, val in enumerate(row):
            if (i > 0 and j > 0):
                table[i][j] = val.center(colWidths[j])
            elif i == 0:
                table[i][j] = val.center(colWidths[j])
            elif j == 0:
                if rowHeader:
                    table[i][j] = val.rjust(colWidths[j])
                else:
                    table[i][j] = val.center(colWidths[j])
            if j != len(table[i]) - 1 and columnSeparator != "":
                table[i][j] += columnSeparator

    out = ""
    for val in table:
        if useSpaces:
            out += "   ".join(val) + "\n"
        else:
            out += "".join(val) + "\n"

    if includeTrailingNewLine:
        return out

    return out.rstrip("\n")
