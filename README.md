# Differentiable user models - Data Generator

This repository contains a modified version of the code used to produce the results of the paper "Differentiable user models". The original version is located [here](https://github.com/hamalajaa/DifferentiableUserModels). In this version (which was not used in the original paper), the synthetic user data is generated and saved to an hdf5 file, but the data are not used to train a model. The main advantage of this repo is that a CUDA-enabled GPU is not required to generate the data.

### Julia environment setup
Users are expected to have some basic Julia knowledge. This is a general workflow for getting the environment set up:  
1. Browse to directory.
2. (Optional) If you are not using Julia 1.6.x on a Mac, remove the Manifest.toml file.
3. Open the Julia REPL. 
4. Open Julia and open the package manager using `]`.  
5. Activate the current directory using `activate .` (note that you need the dot). This tells Julia to use the environment in the current directory.  
6. Instantiate the environment using `instantiate`.
7. Build the environment using `build`.
8. Precompile the vironment using `precompile`.

### Julia environment debugging
It is likely that somewhere in the steps above, there will be errors with one or more packages, as some packages need different versions on different systems. Most issues can be solved with the following steps:
1. Look at the log files and identify the problem packages.
2. In the Julia REPL package manager, remove problem package using `rm [packagename]`.
3. Re-add the problem package using `add [packagename`.
4. Try build and precompile steps above
5. If you have any more errors, you could try remove the `Manifest.toml` file and start again.
6. You could try remove the package and see if it runs without it



### Running the scripts
Once the Julia environment has been set up, the scripts can be run for example using
`julia experiment2_datagen.jl`
The environment will be loaded within the file.

Data will then be saved in the /data/ folder for the relevant experiment.

### Example output HDF file structure
```
filename.hdf
├── metadata
│   ├── gen_type = SearchEnvSampler / menu_search
│   ├── n_users = 19200
│   ├── eval = false
│   ├── n_traj = random(1-8)
│   ├── noise_variance => 1e-8,
│   ├── p_bias = 0.0
└── data
    ├── user_1
    │   ├── xc = d[1]
    │   ├── yc = d[1]
    │   ├── xt = d[1]
    │   ├── yt = d[1]
    ├── user_2
    │   ├── xc = d[1]
    │   ├── yc = d[1]
    │   ├── xt = d[1]
    │   ├── yt = d[1]
    └── ...
```



### Experiment 2
By default, the data are pre-batched into batches of 4 users, with 1-8 trajectories randomly generated on a per-batch basis, for all users within a batch.

