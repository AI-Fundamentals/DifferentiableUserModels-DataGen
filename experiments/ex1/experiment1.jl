using ArgParse
using BSON
using Distributions
using Flux
using Stheno
using Tracker

include("../../NeuralProcesses.jl/src/NeuralProcesses.jl")
using .NeuralProcesses

parser = ArgParseSettings()
@add_arg_table! parser begin
    "--gen"
        help = "Experiment setting: gridworld, menu_search, h_menu_search"
        arg_type = String
        default = "gridworld"
    "--n_traj"
        help = "Number of context trajectories. Setting to 0 randomizes between 1 and 8."
        arg_type = Int
	default = 0
    "--n_epochs"
	help = "Number of training epochs."
        arg_type = Int
        default = 50
    "--n_batches"
        help = "Number of batches."
        arg_type = Int
        default = 100
    "--batch_size"
        help = "Batch size."
        arg_type = Int
        default = 4
    "--bson"
        help = "Directly specify the file to save the model to and load it from."
        arg_type = String
    "--epsilon"
        help = "Value for epsilon."
        arg_type = Float64
end
args = parse_args(parser)

println("Initializing model...")

model = anp_ex1(
    dim_embedding=128,
    num_encoder_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    args=args
) |> gpu

#model = BSON.load("models/ex1/dp/dp_01_50/46.bson")[:model] |> gpu

println("Initializing loss...")

loss(xs...) = np_elbo(
    xs...,
    num_samples=5,
    fixed_σ_epochs=3
)

batch_size  = args["batch_size"]

# Redundant. Required to fit the DataGenerator definition
x_context = Distributions.Uniform(-2, 2)
x_target  = Distributions.Uniform(-2, 2)

num_context = Distributions.DiscreteUniform(50, 50)
num_target  = Distributions.DiscreteUniform(50, 50)

data_gen = NeuralProcesses.DataGenerator(
                MCTSPlanner(args;),
                batch_size=batch_size,
                x_context=x_context,
                x_target=x_target,
                num_context=num_context,
                num_target=num_target,
                σ²=1e-8
            )

println("Proceeding to training loop...")

mkpath("models/"*string(args["bson"]))

train_model!(
        model,
        loss,
        data_gen,
        ADAM(5e-4),
        bson=args["bson"],
	experiment=args["gen"],
        starting_epoch=0,
        tasks_per_epoch=2^7,
        batches=args["n_batches"],
	total_epochs=args["n_epochs"],
        epsilon=args["epsilon"]
    )










