# ElemNet

ElemNet is a deep neural network model that takes only the elemental compositions as inputs and leverages artificial intelligence to automatically capture the essential chemistry to predict materials properties. ElemNet can automatically learn the chemical interactions and similarities between different elements which allows it to even predict the phase diagrams of chemical systems absent from the training dataset more accurately than the conventional machine learning models based on physical attributes levaraging domain knowledge.

<p align="center">
  <img src="images/oqmd_approach.png" width="600">
</p>


This repository contains the code for performing data processing, model training and analysis along with the trained model. If you have a large dataset such as OQMD, the model should be trained from scratch. Otherwise for smaller DFT-computed or experimental datasets, it is best to train the model using trasfer learning from a pretrained model as shown below.

<p align="center">
  <img src="images/ElemNet-TL.png" width="600">
</p>

## Installation Requirements

As this git repository uses submodules, you need to clone it with `git clone --recursive` to gather all of the required source code.

The basic requirement for re-using these environments are a Python 3 Jupyter environment with the packages listed in `requirements.txt`.

Some analyses required the use of [Magpie](https://bitbucket.org/wolverton/magpie), which requires Java JDK 1.7 or greater.
See [the Magpie documentation for details].

## Source Files

The code for training the ElemNet model along with the trained model produced in this work are available in the [`elemnet` folder](./elemnet). The other folders contain scripts associted with different analyses performed to characterize ElemNet. The analysis notebooks should be self-describing, in other case there is a README file that describes the folder content.

* [`chemical-interpolation`](./chemical-interpolation): code for creating training sets for chemical interpolation test to determine whether machine learning models are able to infer the interactions between elements that are not included in the training set by plotting their phase diagrams [1].

* [`elemnet`](./elemnet): code for data preprocessing and training the ElemNet model from scratch [1] and using transfer learning from a pretrained model [2], and also for making predictions using a trained model. It contains a README file that explains the details about training and making predictions using ElemNet model.

* [`evaluate-predictions`](./evaluate-predictions): code for evaluating the predictions made by the ElemNet model [1].

* [`icsd-search`](./icsd-search): list of matched compositions between ICSD and ones from combinatorial screening using ElemNet [1].

* [`layer-projection`](./layer-projection): analysis of the activations from ElemNet to interpret how the model is learning the underlying chemistry.

* [`training-data`](./training-data): the training data use for training ElemNet model in [1] and [2]. We use 'oqmd_all.data' in [1], the respective training and test (validation) sets are train_set_230960.data and test_set.data. For [2], we used oqmd_all-22Mar18.csv (OQMD), jv.csv (JARVIS), mp.csv (the Materials Project) and exp.csv (experimental observations) as the datasets.

## Running the code

The code to run the ElemNet model is privided inside the [`elemnet`](./elemnet) folder. Inside this folder, you can simply run the code by passing a sample config file to the dl_regressors.py as follows:

`python dl_regressors.py --config_file sample/sample-run.config`

The config file defines the loss_type, training_data_path, test_data_path, label, input_type [elements_tl for ElemNet] and other runtime parameters. For transfer learning, you need to set 'model_path' [e.g. `sample/sample_model`]. The output log
from this sample run is provided in the `sample/sample.log` file.



## Developer Team

The code was developed by Dipendra Jha from the <a href="http://cucis.ece.northwestern.edu/">CUCIS</a> group at the Electrical and Computer Engineering Department at Northwestern University and Logan Ward from the Computation Institute at University of Chicago.



## Publications

Please cite the following works if you are using ElemNet model:

1. Dipendra Jha, Logan Ward, Arindam Paul, Wei-keng Liao, Alok Choudhary, Chris Wolverton, and Ankit Agrawal, “ElemNet: Deep Learning the Chemistry of Materials From Only Elemental Composition,” Scientific Reports, 8, Article number: 17593 (2018) [DOI:10.1038/s41598-018-35934-y]  [<a href="https://www.nature.com/articles/s41598-018-35934-y">PDF</a>].

2. Dipendra Jha, Kamal Choudhary, Francesca Tavazza, Wei-keng Liao, Alok Choudhary, Carelyn Campbell, Ankit Agrawal, "Enhancing materials property prediction by leveraging computational and experimental data using deep transfer learning," Nature Communications, 10, Article number: 5316 (2019) [DOI: https:10.1038/s41467-019-13297-w] [<a href="https://www.nature.com/articles/s41467-019-13297-w">PDF</a>].

## Questions/Comments

email: dipendra.jha@eecs.northwestern.edu, loganw@uchicago.edu or ankitag@eecs.northwestern.edu</br>
Copyright (C) 2019, Northwestern University.<br/>
See COPYRIGHT notice in top-level directory.


## Funding Support

This work was performed under the following financial assistance award 70NANB14H012 and 70NANB19H005 from U.S. Department of Commerce, National Institute of Standards and Technology as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from DOE awards DE-SC0014330, DE-SC0019358.
