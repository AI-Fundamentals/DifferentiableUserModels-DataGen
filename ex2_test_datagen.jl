""" Script to generate test data for ex2.jl
"""

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
    function get_default_args()
        defaults = Dict(
            "gen" => "menu_search",
            "n_traj" => 10,
            "n_epochs" => 100,
            "batch_size" => 1,
            "params" => false,
            "p_bias" => 0.0,
            "bson" => "",
            "bson_r" => ""
        )
        return defaults
    end
    
    args = get_default_args()



    # Make the data generator
    println("Initializing data generator")
    
    batch_size  = args["batch_size"]
    
    # Redundant. Required to fit the DataGenerator definition
    x_context = Distributions.Uniform(-2, 2)
    x_target  = Distributions.Uniform(-2, 2)
    
    num_context = Distributions.DiscreteUniform(50, 50)
    num_target  = Distributions.DiscreteUniform(50, 50)
    
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
# all 32 tasks

tasks = 32

# Function to help print output in realtime in jupyter notebooks

println("Starting generating data with $n_workers workers")
data = @distributed (vcat) for task_n in 1:tasks;
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
"tasks" => tasks,
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
filepath = "data/ex2/experiment2_test_data.hdf"
create_hdf5_ex2(data,filepath,metadata)
println("File saved successfully")