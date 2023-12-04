""" Script to generate data for experiment2.jl
The original script generated was quite confused (and confusing) in terms of the number of the number of tasks/batches/minibatches.
Originally the script created 192 tasks split up into 24 batches.
This calculation comes from an input of "2^5 (32) tasks" multiplied by 25 (actually 24) batches, where a batch is an arbitrary way
to split the training dataset into separate files. In this implementation, the training dataset is simply split into one HDF5 file
of 192 tasks, so the minibatches can be dealt with when training the model."""

# Run the script in parallel
using Distributed

# Add processes
rmprocs(workers()) # This will remove all worker processes
n_workers = 8
addprocs(n_workers) # Change this to the number of cores you want to use

@everywhere begin
    # Sort out environment
    using Pkg
    Pkg.activate(".")
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
    batch_size  = args["batch_size"]
    
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
    println("Data gen initialized")

end



# Generate the data in a parallel way. The vector "data" will be the dataset from
# all 192 tasks

tasks_per_epoch = 192

# Function to help print output in realtime in jupyter notebooks

println("Starting generating data with $n_workers workers")
data = @distributed (vcat) for task_n in 1:tasks_per_epoch;
    println("Starting task $task_n")
    flush(stdout)
    
    # Generate data
    data = gen_batch(data_gen, 1; eval=false)
    #data = gen_batch(data_gen, tasks_per_worker; eval=false)
    
    # Return the data from this worker to the "big" data array above
    data;
end

println("Finished generating data")



# Add multiple pieces of metadata to the dataset   
metadata = Dict(
"gen_type" => "SearchEnvSampler / menu_search",
"tasks_per_epoch" => tasks_per_epoch,
"eval" => false,
"batch_size" => batch_size,
"n_traj" => "random(1-8), although this doesn't seem to be used", #This is what happens when it's set to 0 in args dictionary
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
            g = create_group(fid, "task_$i")

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
filepath = "data/ex2/experiment2_data.hdf"
create_hdf5_ex2(data,filepath,metadata)
println("File saved successfully")