pysco
=========

Git repository for developing code to extract, model, and do inference with self-calibrating interferometric observables, including kernel phases, closure phases, bispectra and complex visibilities.

At present, these functions are not all merged: contains both a sub-folder pysco, which is a module that does kernel phase only, and a module pymask, which only does sparse aperture masking. A long term goal is to merge these into a single module that can do both.

This has been built upon https://code.google.com/p/pysco/, written principally by Frantz Martinache, Mike Ireland and on pyker by Frantz Martinache and Benjamin Pope, and includes the deprecated pyscosour additions by Pope and image reconstruction code by Ireland and Greenbaum. This is the product of a collaborative team.

Copyright (C) 2014 Pope, Martinache, Ireland, Cheetham, Greenbaum, Latyshev

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
