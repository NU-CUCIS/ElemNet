This directory contains training and test sets for a 'phase diagram holdout test' generated using create-training-sets.ipynb notebook. To generate the '-fraction.csv' and '-physical.csv', please use the  generate-features.scala (this uses magpie).

There are data for two different holdout tests:
    1. Excluding all of the data from the Na-Mn-Fe-O phase diagram. (Anything that contains exclusively Na, Mn, Fe, and O)
    2. Excluding all pairwise interactions of Ti and O. (Anything that contains both Ti and O)
   
The files denoted '.data' are in a format compatible with Magpie. 

The '-physical.csv' files contain the physics-inspired features of Ward 2016, and the '-fraction.csv' files contain only the element fractions.

The 'NaFeMnO_search-space.csv' file contains a series of generated compositions for the Na-Mn-Fe-O phase diagram. This file is included for plotting the predictions of the model.
