# ElemNet
ElemNet is a 17-layered fully connected network for the prediction of formation energy (enthalpy) from elemental compositions only. This repository contains the model weights and a Jupyter notebook for making predictions using the ElemNet model.

<b>Input</b>: Takes a 2D numpy array with the rows representing different compounds, and columns representing the elemental compositions with 86 elements in the set elements- ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'], elemental compositon does not contain any element from ['He', 'Ne', 'Ar', 'Po', 'At','Rn','Fr','Ra']

<b>Output</b>: Returns a 1D numpy array with the predicted formation energy

<b>Python version</b>: 3.6

<b>Prerequisite softwares</b>: TensorFlow and Numpy (see requirements.txt in root directory)
