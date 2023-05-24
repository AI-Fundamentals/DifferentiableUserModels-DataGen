# Differentially Private Probabilistic User Modeling

This repository contains the code used to produce the results of the paper "Differentially Private Probabilistic User Modeling". 

This code largely utilizes https://github.com/hamalajaa/DifferentiableUserModels as a basis and is built on top of the NeuralProcesses.jl library (https://github.com/wesselb/NeuralProcesses.jl). **We further emphasize that the code included in the NeuralProcesses.jl folder in this project does not represent our contribution** and is only slightly modified for the purposes of this work.

### Running the experiments

The (A)NP model can be trained for the experiment settings introduced in the paper with the following commands:

**Experiment 1 training:**
```
$ julia --project=Project.toml experiments/ex1/experiment1.jl --gen gridworld --n_epochs=100 --n_batches [n] --bson ex1/dp/dp_[e]_[n] --epsilon [e]
```
where `[n]` corresponds to the number user batches (1 batch contains 128 users) and `[e]` controls the epsilon value.

**Experiment 2 training:**
```
$ julia --project=Project.toml experiments/ex2/experiment2.jl --gen menu_search --n_epochs=200 --n_batches [n] --bson ex2/dp/dp_[e]_[n] --epsilon [e]
```
where `[n]` corresponds to the number user batches (1 batch contains 32 users) and `[e]` controls the epsilon value.


After training, the models can be straightforwardly evaluated with:
```
$ julia --project=Project.toml experiments/ex1/ex1_test.jl --gen gridworld --n_epochs=100 --bson dp_[e]_[n] --bson_r results/ex1/dp_[e]_[n].bson
```
and
```
$ julia --project=Project.toml experiments/ex2/ex2_test.jl --gen menu_search --n_epochs=200 --bson dp_[e]_[n] --bson_r results/ex2/dp_[e]_[n].bson
```
