# Differentiable user models - Data Generator

This repository contains a modified version of the code used to produce the results of the paper "Differentiable user models". The original version is located [here](https://github.com/hamalajaa/DifferentiableUserModels). In this version (which was not used in the original paper), the synthetic user data is generated and saved to an hdf5 file, but the data are not used to train a model. The main advantage of this repo is that a CUDA-enabled GPU is not required.

### Julia environment setup
Users are expected to have some basic Julia knowledge. This is a general workflow for getting the environment set up:  
1. Browse to directory.  
2. Remove any Manifest.toml file.  
3. Open Julia and open the package manager using `]`.  
4. Update the package registry using `update`.  
5. Activate the current directory using `activate .` (note that you need the dot). This tells Julia to use the Project.toml file in the current directory.  
6. Instantiate the environment using `instantiate`.

Julia will install and precompile all the required packages.

### Running the scripts
Once the Julia environment has been set up, the scripts can be run for example using
`julia experiment2_datagen.jl`
The environment will be loaded within the file.

Data will then be saved in the /data/ folder for the relevant experiment.

### Output HDF file structure
```
filename.hdf
├── metadata
│   ├── batch_size = 4
│   ├── n_batches = 4800
│   ├── ...
└── data
    ├── batch_1
    │   ├── xc = d[1]
    │   ├── yc = d[1]
    │   ├── xt = d[1]
    │   ├── yt = d[1]
    ├── batch_2
    │   ├── xc = d[1]
    │   ├── yc = d[1]
    │   ├── xt = d[1]
    │   ├── yt = d[1]
    └── ...
```



### Experiment 2
By default, the data are pre-batched into batches of 4 users, with 1-8 trajectories randomly generated on a per-batch basis, for all users within a batch.

