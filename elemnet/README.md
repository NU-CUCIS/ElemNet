# ElemNet Model
ElemNet is a 17-layered fully connected network for the prediction of formation energy (enthalpy) from elemental compositions only. This repository contains the model weights and a Jupyter notebook for making predictions using the ElemNet model.

<b>Input</b>: Takes a 2D numpy array with the rows representing different compounds, and columns representing the elemental compositions with 86 elements in the set elements- ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'], elemental compositon does not contain any element from ['He', 'Ne', 'Ar', 'Po', 'At','Rn','Fr','Ra']

<b>Output</b>: Returns a 1D numpy array with the predicted formation energy

<b>Python version</b>: 3.6

<b>Prerequisite softwares</b>: TensorFlow and Numpy (see requirements.txt in root directory)

## Contents

* `data_preprocess.ipynb`: Jupyter notebook that illustrates how to preprocess data for ElemNet. It uses the oqmd-all-22Mar18.csv dataset as sample, other datasets can be similarly preprocessed. Please run this with the respective data file to create the training and test/validation sets before running the model.

* `dl_regressors.py`: Code to run the ElemNet model for training.

* `data_utils.py`: Utility code for data loading.

* `train_utils.py`: Utility code for training the model.

* `sample`: A sample run folder that contains running configuration and the ElemNet trained using random split of oqmd-all-22Mar18.csv. The 'sample_model' can be used for transfer learning.

## Running the code

You can simply run the code by passing a sample config file to the dl_regressors.py as follows:

`python dl_regressors.py --config_file sample/sample-run.config`

The config file defines the loss_type, training_data_path, test_data_path, label, input_type [elements_tl for ElemNEt] and other runtime parameters. For transfer learning, you need to set 'model_path' [e.g. sample/sample_model].

## Developer Team

The code was developed by Dipendra Jha from the <a href="http://cucis.ece.northwestern.edu/">CUCIS</a> group at the Electrical and Computer Engineering Department at Northwestern University and Logan Ward from the Computation Institute at University of Chicago.



## Citation

Please cite the following works if you are using ElemNet model:

1. D. Jha, L. Ward, A. Paul, W.-keng Liao, A. Choudhary, C. Wolverton, and A. Agrawal, “ElemNet: Deep Learning the Chemistry of Materials From Only Elemental Composition,” Scientific Reports, 8, Article number: 17593 (2018) [DOI:10.1038/s41598-018-35934-y]  [<a href="https://www.nature.com/articles/s41598-018-35934-y">PDF</a>].

2. D. Jha, K. Choudhary, F. Tavazza, W.-keng Liao, A. Choudhary, C. Campbell, A. Agrawal, "Enhancing materials property prediction by leveraging computational and experimental data using deep transfer learning," Nature Communications, 10, Article number: 5316 (2019) [DOI: https:10.1038/s41467-019-13297-w] [<a href="https://www.nature.com/articles/s41467-019-13297-w">PDF</a>].

## Questions/Comments

email: dipendra.jha@eecs.northwestern.edu, or ankitag@eecs.northwestern.edu</br>
Copyright (C) 2019, Northwestern University.<br/>
See COPYRIGHT notice in top-level directory.


## Funding Support

This work was performed under the following financial assistance award 70NANB14H012 and 70NANB19H005 from U.S. Department of Commerce, National Institute of Standards and Technology as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from DOE awards DE-SC0014330, DE-SC0019358.
