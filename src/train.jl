export eval_model!, train_model!

"""
Essentially the same file as src/expeirment/experiment.jl in NeuralProcesses.jl with slight modifications.
"""

using .NeuralProcesses

using BSON
#using CUDA
using Flux
using Printf
using ProgressMeter
using Stheno
using Tracker
using Statistics

import StatsBase: std

include("../NeuralProcesses.jl/src/experiment/checkpoint.jl")

function eval_model!(model, loss, data_gen, epoch::Integer; num_batches::Integer=1)
    model = NeuralProcesses.untrack(model)
    
    # Generate evaluation data
    @time data = gen_batch(data_gen, num_batches)

    tuples = map(x -> loss(model, epoch, gpu.(x)...), data)
    values = map(x -> x[1], tuples)
    sizes = map(x -> x[2], tuples)

    # Compute and print loss.
    loss_value, loss_error = _mean_error(values)
    println("Losses:")
    @printf(
        "    %8.3f +- %7.3f (%d batches)\n",
        loss_value,
        loss_error,
        num_batches
    )

    # Normalise by average size of target set.
    @printf(
        "    %8.3f +- %7.3f (%d batches; normalised)\n",
        _mean_error(values ./ mean(sizes))...,
        num_batches
    )

    # Normalise by the target set size.
    @printf(
        "    %8.3f +- %7.3f (%d batches; global mean)\n",
        _mean_error(values ./ sizes)...,
        num_batches
    )

    likelihoods(xs...) = likelihood(
        xs...,
        target=true,
        num_samples=5,
        fixed_σ_batches=3
    )
    
    tuples = map(x -> likelihoods(model, epoch, gpu.(x)...), data)
    values = map(x -> x[1], tuples)
    sizes  = map(x -> x[2], tuples)

    # Compute and print loss
    lik_value, lik_error = _mean_error(values)
    println("Likelihoods of observations:")
    @printf(
        "    %8.3f +- %7.3f (%d batches)\n",
        lik_value,
        lik_error,
        num_batches
    )

    # Normalise by average size of target set.
    @printf(
        "    %8.3f +- %7.3f (%d batches; normalised)\n",
        _mean_error(values ./ mean(sizes))...,
        num_batches
    )

    # Normalise by the target set size.
    @printf(
        "    %8.3f +- %7.3f (%d batches; global mean)\n",
        _mean_error(values ./ sizes)...,
        num_batches
    )

    return loss_value, loss_error, lik_value, lik_error
end


_mean_error(xs) = (Statistics.mean(xs), 2std(xs) / sqrt(length(xs)))

_nanreport = Flux.throttle(() -> println("Encountered NaN loss! Returning zero."), 1)

function _nansafe(loss, xs...)
    value, value_size = loss(xs...)
    if isnan(value) || abs(value) > 1000000
        _nanreport()
        return Tracker.track(identity, 0f0), value_size
    else
        return value, value_size
    end
end

function train_model!(
    model,
    loss,
    data_gen,
    opt;
    bson=nothing,
    experiment::String,
    starting_epoch::Integer=1,
    batches::Integer=100,
    tasks_per_epoch::Integer=1000,
    total_epochs::Integer=50,
    epsilon::Float64=1.
)
    CUDA.GPUArrays.allowscalar(false)
    
    # Divide out batch size to get the number of batches per epoch.
    batches_per_epoch = div(tasks_per_epoch, data_gen.batch_size)

    # Display the settings of the training run.
    @printf("Epochs:               %-6d\n", total_epochs)
    @printf("Starting epoch:       %-6d\n", starting_epoch)
    @printf("Tasks per epoch:      %-6d\n", batches_per_epoch * data_gen.batch_size)
    @printf("Batch size:           %-6d\n", data_gen.batch_size)
    @printf("Number of batches     %-6d\n", batches)

    # Track the parameters of the model for training.
    model = NeuralProcesses.track(model)

    loss_means  = []
    loss_errors = []
    lik_means   = []
    lik_errors  = []

    # Clipping bound (<4)
    c = 10.
    L = batches
    
    U = tasks_per_epoch * batches			
    eps = epsilon
    del = 1. / U^2

    sig = get_sigma(eps, del, total_epochs * batches * batches_per_epoch, 1. / (batches * batches_per_epoch))

    
    for batch_n in 1:batches-1
        # Warmup epoch
        if batch_n == starting_epoch
            n_mini_batches = 1
        else
            n_mini_batches = batches_per_epoch
        end
        # Generate data
        data = gen_batch(data_gen, n_mini_batches; eval=false)
	
	if experiment == "gridworld"
            BSON.bson("data/ex1/"*string(batch_n)*".bson", data=data)
	end
	if experiment == "menu_search"
	    BSON.bson("data/ex2/"*string(batch_n)*".bson", data=data)
	end

    end

    for epoch in starting_epoch:total_epochs

        for batch_n in 1:batches-1
            # Perform epoch.
            CUDA.reclaim()
            @time begin
                ps = Flux.Params(Flux.params(model))
            
	    if experiment == "gridworld"
                data = BSON.load("data/ex1/"*string(batch_n)*".bson")[:data]
            end
	    if experiment == "menu_search"
	        data = BSON.load("data/ex2/"*string(batch_n)*".bson")[:data]
	    end

                @showprogress "Epoch $batch_n: " for d in data
                    gs = Tracker.gradient(ps) do
                        first(_nansafe(loss, model, batch_n, gpu.(d)...))
                    end
                    for p in ps
                        g = Tracker.data(gs[p])
                        g = min.(1., c ./ LinearAlgebra.norm(g)) .* g
                        n = CUDA.randn(Float64, size(g)) .* sqrt(sig^2 .* c^2) |> gpu
                        g = g .+ n
                        Tracker.update!(p, -Flux.Optimise.apply!(opt, Tracker.data(p), g))
                    end
                end
            end

            # Evalute model.
            CUDA.reclaim()
            loss_value, loss_error, lik_value, lik_error = eval_model!(
                NeuralProcesses.untrack(model),
                loss,
                data_gen,
                batch_n
            )

            push!(loss_means,  loss_value)
            push!(loss_errors, loss_error)
            push!(lik_means,   lik_value)
            push!(lik_errors,  lik_error)

            CUDA.reclaim()

            # Save result

	    BSON.bson("models/"*bson*"/"*string(epoch)*".bson", model=NeuralProcesses.untrack(model))
	    """
            if !isnothing(bson)
                checkpoint!(
                    "models/"*bson*"/"*string(epoch)*".bson",
                    NeuralProcesses.untrack(model),
                    batch_n,
                    loss_value,
                    loss_error
                )
            end
	    """
        end

	mkpath("results/"*bson)

        BSON.bson("results/"*bson*".bson", loss_means=loss_means, loss_stds=loss_errors,
                                    lik_means=lik_means, lik_stds=lik_errors)

    end

end
