""" Script to generate data for experiment2.jl
The original script generated was quite confused (and confusing) in terms of the number of the number of tasks/batches/minibatches.
Originally the script created 768 tasks split up into 24 batches. It was supposed to be 800 from 25 batches but there was a typo.
This calculation comes from an input of "2^5 (32) tasks_per_epoch" multiplied by 25 (actually 24) batches, where a batch is an arbitrary way
to split the training dataset into separate files.
In this implementation here, the training dataset is simply split into one HDF5 file 192 tasks, so the minibatches can be dealt with when training the model."""

# Run the script in parallel
using Distributed

## Add processes

# Get the number of available cores
n_cores = Sys.CPU_THREADS
# Leave one core for operating system. Comment this out if running on dedicated computing infrastructure
n_workers = n_cores-1
(println("Setting up to run with $n_workers workers"),flush(stdout))

# Remove all worker processes
rmprocs(workers())

# Add processes equal to the number of available cores minus one
addprocs(n_workers)


# The number of users to model
n_users = 19200

@everywhere begin
    # Sort out environment
    using Pkg
    Pkg.activate(".")
    (println("Activated environment"),flush(stdout))
    Pkg.instantiate()
end

@everywhere begin
    using ArgParse
    using BSON
    using Distributions
    using Flux
    using Stheno
    using Tracker
    using Printf
    using HDF5
    using SharedArrays
end

# Add neuralprocesses package
@everywhere include(joinpath(@__DIR__, "NeuralProcesses.jl/src/NeuralProcesses.jl"))
@everywhere using .NeuralProcesses
    
        
@everywhere begin

    # Make a dictionary to just use the default arguments from the argument parser
    # Not all of these are used in this script
    function get_default_args()
        defaults = Dict(
            "gen" => "menu_search",
            "n_traj" => 0,
            "n_epochs" => 50,
            "n_batches" => 25,
            "batch_size" => 4,
            "params" => false,
            "p_bias" => 0.0,
            "bson" => "",
            "epsilon" => 0.0
        )
        return defaults
    end
    
    args = get_default_args()


    # Build the DataGenerator
    println("Initializing data generator")
    flush(stdout)
    # We make the batch size 1 here, and batch the data when loading and training the model
    batch_size  = 1
    
    # Redundant. Required to fit the DataGenerator definition
    x_context = Distributions.Uniform(-2, 2)
    x_target  = Distributions.Uniform(-2, 2)
    
    num_context = Distributions.DiscreteUniform(10, 10)
    num_target  = Distributions.DiscreteUniform(10, 10)
    
    data_gen = NeuralProcesses.DataGenerator(
                    SearchEnvSampler(args;),
                    batch_size=batch_size,
                    x_context=x_context,
                    x_target=x_target,
                    num_context=num_context,
                    num_target=num_target,
                    σ²=1e-8
                )
    (println("Data gen initialized"),flush(stdout))

end



# Generate the data in a parallel way. The vector "data" will be the dataset from all users
(println("Starting generating data with $n_workers workers"),flush(stdout))
data = @distributed (vcat) for user_n in 1:n_users;
    (println("Starting task $user_n"),flush(stdout))
    
    # Generate data
    data = gen_batch(data_gen, 1; eval=false)
    #data = gen_batch(data_gen, tasks_per_worker; eval=false)
    
    # Return the data from this worker to the "big" data array above
    data;
end

(println("Finished generating data"),flush(stdout))



# Add multiple pieces of metadata to the dataset   
metadata = Dict(
"gen_type" => "SearchEnvSampler / menu_search",
"n_users" => n_users,
"eval" => false,
"n_traj" => "random(1-8)", #This is what happens when it's set to 0 in args dictionary
"noise_variance" => 1e-8,
"p_bias" => 0.0
)

# Function to save the data as HDF5
function create_hdf5_ex2(data, filename, metadata)
    # Open the HDF5 file for writing, overwriting if it exists
    h5open(filename, "w") do fid
        # Loop over the data vector
        for (i, d) in enumerate(data)
            # Create a group for each mini-batch
            g = create_group(fid, "user_$i")

            # Add datasets to the group
            g["xc"] = d[1]
            g["yc"] = d[2]
            g["xt"] = d[3]
            g["yt"] = d[4]

            # Add metadata to the group
            for (key, value) in metadata
                write_attribute(g, key, value)
            end
        end
    end
end


# Save the data!
folderpath = "data/ex2/"
filepath = folderpath * "experiment2_data.hdf"

if !isdir(folderpath)
    mkpath(folderpath)
end

create_hdf5_ex2(data,filepath,metadata)
(println("File saved successfully"),flush(stdout))