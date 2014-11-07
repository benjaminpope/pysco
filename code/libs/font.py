# Written by Guillaume, 2014

"""
This module provides the user with a set of string constants to be concatenated, in order to add colors and/or formatting to terminal display.

>>> import misc.font as font
>>> print 'Hello, '+font.red+font.underlined+' I am in underlined red'+font.normal

>>> font. <tab>
Will give you the list of available colors and formatting options

Colors and font attributes taken from: http://misc.flogisoft.com/bash/tip_colors_and_formatting
"""

white = '\033[97m'
black = '\033[38;5;16m'
gray = '\033[90m'
red = '\033[31m'
green = '\033[32m'
yellow = '\033[33m'
orange = '\033[38;5;166m'
blue = '\033[34m'
magenta = '\033[35m'

nocolor = '\033[39m'

bold = '\033[1m'
nobold = '\033[21m'

underlined = '\033[4m'
nounderlined = '\033[24m'

dim = '\033[2m'
nodim = '\033[22m'

normal = nodim + nobold + nobold + nocolor
