import copy
import math


class TableError(Exception):
    pass


def objectClass(obj):
    "returns a string representing the class of the given object"
    try:
        #return str(type(obj)).split("'")[1]
        return str(obj.__class__).split(".")[1].replace("<type '", "").replace("'>", "")
    except IndexError:
        return str(type(obj)).replace("<type '", "").replace("'>", "")
    #raise Errors.Badness("Could not get the class of object " + str(obj) + " with type " + str(type(obj)) )


def tableString(table, rowHeader=True, headers=None, roundDigits=None, columnSeperator="", sortColumn=None,
                maxRowsToShow=None, snipIndex=None, useSpaces=True, includeTrailingNewLine=True):
    """takes a table (rows and columns of strings) and returns a string representing a nice visual representation of that table.
    roundDigits is the number of digits to round floats to"""
    if not isinstance(table, (list, tuple)): raise TableError("table must be list or tuple")
    if len(table) == 0: return ""
    if not isinstance(table[0], (list, tuple)): raise TableError(
        "table elements be lists or tuples. You gave: " + str(objectClass(table[0])))

    table = copy.deepcopy(table)    #So that the table doesn't get destroyed in the process!
    colWidths = []
    #rows = len(table)
    cols = 0
    for row in table:
        if not isinstance(row, list): raise TableError(
            "table must be a list of lists but found a row that had the value " + str(row))
        if (len(row) > cols): cols = len(row)

    for c in xrange(cols):
        colWidths.append(1)

    #sort the values if sorting is on
    if sortColumn != None:
        if isinstance(sortColumn, (str, unicode)): #if we're sorting by the column with a given name
            if headers == None: raise Exception(
                "Cannot find the sortColumn '" + str(sortColumn) + "' since headers=" + str(headers))
            sortColumn = headers.index(sortColumn)
        if not isinstance(sortColumn, (int, long)): raise Exception(
            "sort column must be an integer, but was: " + str(sortColumn))
        if sortColumn < 0: raise Exception(
            "sortColumn should have been a non-negative integer but was: " + str(sortColumn))
        if table[0] == headers:
            tempHeaders = table[0]
            table = table[1:]
        else:
            tempHeaders = None
        table.sort(lambda x, y: cmp(y[sortColumn], x[sortColumn])) #sort decreasing by specified column
        if tempHeaders != None:
            table.insert(0, tempHeaders)

    #replace numbers with formatting/rounding versions and update column widths to fit the values
    for r in xrange(len(table)):
        for c in xrange(len(table[r])):
            if roundDigits != None and isinstance(table[r][c], float):
                #print "table[r][c]", table[r][c]
                #print "roundDigits", roundDigits
                table[r][c] = formatNumber(table[r][c], roundDigits)
            else:
                table[r][c] = str(table[r][c])
            if (len(table[r][c]) > colWidths[c]): colWidths[c] = len(table[r][c])

    if headers != None:
        if len(headers) != cols: raise TableError(
            "Number of table columns (" + str(cols) + ")  does not match number of header columns (" + str(
                len(headers)) + ")!")
        for c in xrange(len(headers)):
            if colWidths[c] < len(headers[c]): colWidths[c] = len(headers[c])

    #if there is a limit to how many rows we can show, delete the middle rows and replace them with a "..." row
    if maxRowsToShow != None:
        numToDelete = max(len(table) - maxRowsToShow, 0)
        if numToDelete > 0:
            firstToDelete = int(math.ceil((len(table) / 2.0) - (numToDelete / 2.0)))
            lastToDelete = firstToDelete + numToDelete - 1
            table = table[:firstToDelete] + [["..."] * len(table[firstToDelete])] + table[lastToDelete + 1:]
    #if we want to imply the existence of more rows, but they are not currently present in the table, so
    #we just add an elipses at the specified index
    elif snipIndex != None and snipIndex > 0:
        table = table[:snipIndex] + [["..."] * len(table[firstToDelete])] + table[snipIndex + 1:]

    #modify the text in each column to give it the right length
    for r in xrange(len(table)):
        for c in xrange(len(table[r])):
            v = table[r][c]
            if (r > 0 and c > 0):
                table[r][c] = v.center(colWidths[c])
            elif (r == 0):
                table[r][c] = v.center(colWidths[c])
            elif (c == 0):
                if (rowHeader):
                    table[r][c] = v.rjust(colWidths[c])
                else:
                    table[r][c] = v.center(colWidths[c])
            if c != len(table[r]) - 1 and columnSeperator != "":
                table[r][c] += columnSeperator

    out = ""
    for r in xrange(len(table)):
        if useSpaces:
            out += "   ".join(table[r]) + "\n"
        else:
            out += "".join(table[r]) + "\n"

    if includeTrailingNewLine:
        return out
    else:
        return out.rstrip("\n")


#the below function is from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/473872
def formatNumber(num, places=0):
    """Format a number with grouped thousands and given decimal places"""
    if not isinstance(num, float): return str(num)
    if num < 0:
        wasNeg = True
        num *= -1
    else:
        wasNeg = False
    #print "Places:",places, type(places), type(max)
    places = max(0, places)
    #print "New pLaces:",places
    tmp = "%.*f" % (places, num)
    point = tmp.find(".")
    integer = (point == -1) and tmp or tmp[:point]
    decimal = (point != -1) and tmp[point:] or ""
    count = 0
    formatted = []
    #print integer, type(integer)
    #print len(integer)
    for i in xrange(len(integer), 0, -1):
        count += 1
        formatted.append(integer[i - 1])
        if count % 3 == 0 and i - 1:
            formatted.append(",")
    integer = "".join(formatted[::-1])
    out = integer + decimal
    if wasNeg: return "-" + out
    return out