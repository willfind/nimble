
import sys
import os
import inspect
import json

try:
    import clang
    import clang.cindex
    clangPresent = True
except:
    clangPresent = False


OUTFILE_PREFIX = 'shogunParameterManifest_v'
METADATA_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def discoverConstructors(path, desiredFile=None, desiredExt=['.cpp']):
    """
    Recursively visit all directories in the given path, calling
    findConstructors for each cpp source file

    """
    results = {}
    contents = []
    for (folderPath, subFolders, contents) in os.walk(path):
        for fileName in contents:

            # Required for shogun 6.0.0, 5.0.0, 4.1.0, 4.0.0, 3.2.1, 3.2.0, 3.1.1, 3.1.0, 3.0.0
#            if fileName in ['SGVector.cpp']:
#                continue
            filePath = os.path.join(folderPath, fileName)
            (rootName, ext) = os.path.splitext(fileName)
            (rootPath, ext) = os.path.splitext(filePath)
            if desiredFile is None or rootName in desiredFile:
                if ext in desiredExt:
                    print (filePath)
                    findConstructors(filePath, results, rootPath)
                        

#    pdb.set_trace()

    print (len(results))

    return results


def findConstructors(fileName, results, targetDirectory):
    """ Find all constructors and list their params in the given file """
    index = clang.cindex.Index.create()
    parsed = index.parse(fileName)
    tuNode = parsed.cursor
    findConstructorsBackend(tuNode, results, targetDirectory)


def findConstructorsBackend(node, results, targetDirectory):
    """ Recursively visit all nodes, checking if it is a constructor """
    if node.location.file is not None:
        if not node.location.file.name.startswith(targetDirectory):
            return

    if node.kind == clang.cindex.CursorKind.CONSTRUCTOR:
        constructorName = node.spelling
        args = []
        for value in node.get_arguments():
            args.append(value.spelling)
        # TODO value.type.spelling

#        print "%s%s" % (constructorName, str(args))
        if not constructorName in results:
            results[constructorName] = []
        if args not in results[constructorName]:
            results[constructorName].append(args)
    # Recurse for children of this node if it isn't a constructor
    else:
        for child in node.get_children():
            findConstructorsBackend(child, results, targetDirectory)


if __name__ == '__main__':
    # clang is required for this script, but since it is loaded by the test suite
    # we hide the imports so as to not cause any failures and a system without
    # clang. The imports are preproduced here to trigger failures when the script
    # is actually used.
    if not clangPresent:
        import clang
        import clang.cindex

    libclang_location = sys.argv[1]  # path to exact libclang file
    shogunSource_location = sys.argv[2]  # path to folder containing shogun source files
    shogunSource_version = sys.argv[3]  # will be used as is when outputing the metadata file

    # Attempt setup for clang; verifying that we can actually run discovery.
    clang.cindex.Config.set_library_file(libclang_location)
    clang.cindex.Index.create()

    paramsManifest = discoverConstructors(shogunSource_location)

    if len(paramsManifest) != 0:
        writePath = os.path.join(METADATA_PATH, ((OUTFILE_PREFIX + '%s') % shogunSource_version))
        with open(writePath, 'w') as fp:
            json.dump(paramsManifest, fp, indent=4)
