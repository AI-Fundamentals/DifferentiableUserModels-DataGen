""" Script to generate training data for experiment 2.
The original script generated was quite (and confusing) in terms of the number of the number of tasks/batches/minibatches.
Originally the script created 768 tasks split up into 24 batches. It was supposed to be 800 from 25 batches but there was a typo.
This calculation comes from an input of "2^5 (32) tasks_per_epoch" multiplied by 25 (actually 24) batches, where a batch is an arbitrary way
to split the training dataset into separate files.
In this implementation here, the dataset is simply split into one HDF5 file of users (no batches), so the minibatches can be dealt with when training the model.
By default the number of users is set to 64 for testing. To change the number of users, edit the "n_users" parameter."""


# Run the script in parallel
using Distributed

## Add processes

# Get the number of available cores
if haskey(ENV, "NSLOTS")
    n_cores = parse(Int, ENV["NSLOTS"])
    println("From SGE, $(n_cores) total cores.")
else
    n_cores = Sys.CPU_THREADS
    println("From local system, $(n_cores) total cores.")
end


# Leave one core for operating system. Comment this out if running on dedicated computing infrastructure
n_workers = n_cores-1
(println("Setting up to run with $n_workers workers"),flush(stdout))

# Remove all worker processes
rmprocs(workers())

# Add processes equal to the number of available cores minus one
addprocs(n_workers)
flush(stdout)

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
    
    # Edit this to change the number of users
    n_users = SharedArray{Int64}(1)
    n_users[1] = 19200
    
    # Redundant. Required to fit the DataGenerator definition
    x_context = Distributions.Uniform(-2, 2)
    x_target  = Distributions.Uniform(-2, 2)
    
    num_context = Distributions.DiscreteUniform(10, 10)
    num_target  = Distributions.DiscreteUniform(10, 10)
    
    data_gen = NeuralProcesses.DataGenerator(
                    SearchEnvSampler(args;),
                    batch_size=1,
                    x_context=x_context,
                    x_target=x_target,
                    num_context=num_context,
                    num_target=num_target,
                    σ²=1e-8
                )
    (println("Data gen initialized"),flush(stdout))

end

(println("Generating data for $(n_users[1]) users."))


# Generate the data in a parallel way. The vector "data" will be the dataset from all users
(println("Starting generating data with $n_workers workers"),flush(stdout))
data = @distributed (vcat) for user_n in 1:n_users[1];
    (println("Starting batch $user_n"),flush(stdout))
    
    # Generate data
    data = gen_batch(data_gen, 1; eval=false)

    # Swap the dimensions of yc and yt so they work better in the neural process model
    xc, yc, xt, yt = data[1]
    yc = permutedims(yc, [2,1,3])
    yt = permutedims(yt, [2,1,3])
    data[1] = [xc,yc,xt,yt]

    # Return the data from this worker to the "big" data array above
    data;
end

(println("Finished generating data"),flush(stdout))



# Add multiple pieces of metadata to the dataset   
metadata = Dict(
"gen_type" => "SearchEnvSampler / menu_search",
"n_users" => n_users[1],
"eval" => false,
"n_traj" => "random(1-8)", #This is what happens when it's set to 0 in args dictionary
"noise_variance" => 1e-8,
"p_bias" => 0.0
)

# Function to save the data as HDF5
function create_hdf5_ex2(data, filename, metadata)
    # Open the HDF5 file for writing, overwriting if it exists
    h5open(filename, "w") do fid
        # Make one group for metadata
        metadata_group = create_group(fid, "metadata")
        
        # Add metadata to the group
        for (key, value) in metadata
            write_attribute(metadata_group, key, value)
        end

        # Make a second group for data
        data_group = create_group(fid, "data")

        # Loop over the data vector
        for (i, d) in enumerate(data)
            subgroup = create_group(data_group, "user_$i")
            # Create a group for each mini-batch
            # Add datasets to the group
            subgroup["xc"] = d[1]
            subgroup["yc"] = d[2]
            subgroup["xt"] = d[3]
            subgroup["yt"] = d[4]
        end
    end
end


# Save the data!
folderpath = "data/ex2/"
filepath = folderpath * "ex2_train_data.hdf"

if !isdir(folderpath)
    mkpath(folderpath)
end

create_hdf5_ex2(data,filepath,metadata)
(println("File saved successfully"),flush(stdout))