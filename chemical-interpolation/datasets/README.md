This directory contains training and test sets for a 'phase diagram holdout test'. 

The training set files (which contain 'train_set' in the name) contain all entries from the OQMD except those from the Na-Mn-Fe-O phase diagram. 
The test set files contain the entries form the Na-Mn-Fe-O phase diagram, which includes all constituent ternary and binary phase diagrams. For example, NaFeO3 and Fe2Na are included in this phase diagram.

The files denoted '.data' are in a format compatible with Magpie. The '-physical.csv' files contain the physics-inspired features of Ward 2016, and the '-fraction.csv' files contain only the element fractions.

The 'NaFeMnO_search-space.csv' file contains a series of generated compositions for the Na-Mn-Fe-O phase diagram. This file is included for plotting the predictions of the model.