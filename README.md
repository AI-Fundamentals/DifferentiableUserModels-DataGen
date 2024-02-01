# Differentiable user models - Data Generator

This repository contains a modified version of the code used to produce the results of the paper "Differentiable user models". The original version is located [here](https://github.com/hamalajaa/DifferentiableUserModels). In this version (which was not used in the original paper), the synthetic user data is generated and saved to an hdf5 file, but the data are not used to train a model. The main advantage of this repo is that a CUDA-enabled GPU is not required.

### Running the scripts

Once a Julia environment has been set up in the standard way, the scripts can be run by just running for example
`julia experiment2_datagen.jl`

Data will then be saved in the /data/ folder.