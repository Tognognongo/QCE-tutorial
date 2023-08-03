#!/bin/bash

conda env remove -n qnlp-tut
conda env create -f environment.yml
source activate qnlp-tut
conda install -c conda-forge mpi4py openmpi
python3 -m ipykernel install --user --name=qnlp-tut
