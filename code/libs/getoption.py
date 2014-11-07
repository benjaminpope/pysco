# -*- coding: utf-8 -*-
# Written by Guillaume, 2014
# reference: http://www.tutorialspoint.com/python/python_command_line_arguments.htm

"""
This module wraps getopt library to return a convenient object whose properties are lists containing the attributes and options provided in the command-line call.
"""

from getopt import getopt as _getoptgetopt
from misc.funcs import aslist as _miscfuncsaslist
import libs.font as _font

class getoption:
    """
    Wraps getopt library to return a convenient object whose properties are lists containing the attributes and options provided in the command-line call.

    :param sysargv: the command-line call, as it is provided by sys.argv
    :type sysargv: list of string
    :param opt: the concatenation of all authorized one-char options i.e. "-e"
    :type opt: string
    :param optval: the concatenation of all authorized one-char options with input value i.e. "-e 3"
    :type optval: string
    :param param: the list of all authorized multi-char options i.e. "--check"
    :type param: list of string
    :param paramval: the list of all authorized multi-char options with input value i.e. "--name=test"
    :type paramval: list of string
    :param man: the manual page of the script
    :type man: string
    :returns: a python object whose attributes are lists of options and attributes, see note below for details
    :raises: Exception, if the getopt library cannot parse properly the command-line call

    .. note::
        
        The returned object possesses as attributes:
            * ``command``: a string that shows the full command-line
            * ``nargs``: an integer showing the total number of arguments provided
            * ``optkey``: a list of all options provided in the command-line call, in the order of the command-line call
            * ``opt``: a dictionnary of the option provided and their associated value. Value-less options return True
            * ``attrs``: a list of all the attributes provided after the options
            * ``man``: a string that contains the manual of how to use this script
    
    >>> import getoption, sys
    >>> scriptcall = getoption.getoption(sys.argv, opt='wes', optval='r',
        man='This is the manual page. Thanks you for using getoption')
    >>> if scriptcall['r'] is not None: print "r option was given"
    >>> if 'h' in scriptcall.optkey:
    >>>     print scriptcall.man
    >>>     sys.exit()

    .. note::
        
        outputObject['option_name'] will return the value of the option ``option_name`` (or ``True`` is the option has no value), or ``None`` if the ``option_name`` was not provided in the command line call
        
        The option ``-h`` is automatically added to the list of one-char authorized options
        
        The syntax of the script is automatically generated from authorized input given to getoption and added at the beginning of the provided manual ``man``
        
        This module uses :mod:`misc.font` and :func:`misc.funcs.aslist`

    >>> from getoption import getoption
    >>> thecall = "conex -w --logfile=~/log.txt 1 TP 234".split(' ')
    >>> # the .split(' ') is use to emulate the parsing of sys.argv
    >>> scriptcall = getoption(thecall, opt='wes', paramval=['logfile'],
        man='Manual page of that script')
    >>> scriptcall. <tab>
    a.attrs    a.command  a.man      a.nargs    a.opt      a.optkey
    >>> print scriptcall.nargs
    5
    >>> print scriptcall.optkey
    ['w', 'logfile']
    >>> print scriptcall.attrs
    ['1', 'TP', '234']
    >>> print scriptcall['w']
    True
    >>> print scriptcall['logfile']
    ~/log.txt
    >>> print scriptcall['e']
    None
    
    """
    def __init__(self, sysargv, opt='', optval='', param=[], paramval=[], man=""):
        param = _miscfuncsaslist(param)
        paramval = _miscfuncsaslist(paramval)
        self.nargs = len(sysargv[1:])
        self.command = ' '.join(sysargv)
        self._shortOpts = opt + 'h' + ''.join([i+':' for i in optval])
        self._longOpts = param + [i+'=' for i in paramval]
        self.man = 'Syntax:\n' + _font.red + str(sysargv[0]) + _font.blue + "".join([' -'+str(i) for i in opt]) + _font.blue.join([' -'+str(i)+ " "+_font.normal+"<val>" for i in optval]) + _font.blue + "".join([' --'+str(i) for i in param]) + _font.blue.join([' --'+str(i)+_font.normal+"=<val>" for i in paramval])
        try:
            opt, self.attrs = _getoptgetopt(sysargv[1:], self._shortOpts, self._longOpts)
        except:
            print self.man
            raise Exception, "Call error"
        self.man +=  (man!="")*"\n\n" + _font.normal + man
        self.opt = {}
        self.optkey = []
        for element in opt:
            self.optkey.append(element[0].replace('-',''))
            if element[1]=='':
                self.opt.update({element[0].replace('-',''): True})
            else:
                self.opt.update({element[0].replace('-',''): element[1]})

    def __getitem__(self, key):
        if self.opt.has_key(key):
            return self.opt[key]
        else: return None
