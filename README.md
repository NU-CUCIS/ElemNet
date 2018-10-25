# ElemNet

This repository contains the analysis codes and deep learning models associated with "ElemNet: Deep Learning the Chemistry of Materials FromOnly Elemental Composition" by D. Jha et al. 

## Contents

The deep learning model produced in this work is available in the [`elemnet` folder](./elemnet).

The other folders contain scripts associted with different analyses performed to characterize ElemNet. Each folder contains a README file that describes what the analyses are, and the notebooks should be self-describing.

## Installation Requirements

As this git repository uses submodules, you need to clone it with `git clone --recursive` to gather all of the required source code. 

The basic requirement for re-using these environments are a Python 3 Jupyter environment with the packages listed in `requirements.txt`. 

Some analyses required the use of [Magpie](https://bitbucket.org/wolverton/magpie), which requires Java JDK 1.7 or greater. 
See [the Magpie documentation for details]
