whisky
=========

Git repository for developing code to extract, model, and do inference with self-calibrating interferometric observables, including kernel phases, closure phases, bispectra and complex visibilities.

At present, these functions are not all merged: contains both a sub-folder whisky, which is a module that does kernel phase only, and a module pymask, which only does sparse aperture masking. A long term goal is to merge these into a single module that can do both.

This has been built upon https://code.google.com/p/pysco/, written principally by Frantz Martinache, Mike Ireland and on pyker by Frantz Martinache and Benjamin Pope, and includes the deprecated pyscosour additions by Benjamin Pope. This is the product of a collaborative team.

We do not make this software available for commercial purposes.