"""
This is some introductionary text that is here for the whole module.

**This module shows examples how to write good docstrings and make links between modules and functions**
"""

def write_docstrings(param1, param2="haha"):
    """
    This function does something.

    *I have a docstring*, but **won't** be imported if you don't imported it.

    :param param1: A first parameter.
    :type param1: str.
    :param param2: Another parameter, depending on ``param1``
    :type param2: bool.
    :returns: int -- the return code.
    :raises: AttributeError, KeyError

    You can use that magic function if:
      * you are awesome
      * your are really awesome
      * you hate IDL

    You never call this class before calling :func:`another_func`.

    .. note::

       An example of intersphinx is this: you **cannot** use :mod:`misc` on this class.

    >>> print 'this is some code sample followed by the result'
    'this is some code sample followed by the result'

    source: http://codeandchaos.wordpress.com/2012/07/30/sphinx-autodoc-tutorial-for-dummies/
    
    and: https://pythonhosted.org/an_example_pypi_project/sphinx.html#full-code-example
    """
    return None

def another_func():
    """
    Just for fun
    """
    return None
