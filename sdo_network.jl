using UnicodePlots
using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
using SpikeSynchrony
using Statistics
using JLD
using Distributed
using SharedArrays
using Plots
using UnicodePlots
using Evolutionary
using Distributions
using LightGraphs
#using Metaheuristics

using SpikingNN
#using StatsBase
using Random
using SparseArrays
using Revise
##
# Override to function to include a state.
##
SNN.@load_units
unicodeplots()

###
# Network 1.
###


const Ne = 200;
const Ni = 50


# https://github.com/RainerEngelken/JuliaCon2017/blob/master/code/generating_large_random_network_topology.ipynb
# needs less memory (n*k instead of n^2)
function gensparsetopo(n, k, seed)
    # seed random
    #srand(seed)
    p = k / (n - 1)
    #A = sprand(Bool,n,n,p)
    #weights = rand(Uniform(n,k),n,n)

end
#gensparsetopo (generic function with 1 method)
# https://gist.github.com/flcong/2eba0189d7d3686ea9633a6d14398931
const seed = 10
const k = 0.5
const this_size = 50
#const Ground_sparse = gensparsetopo(size,k,seed)

using JLD
using Logging
#const ground_weights
function get_constant_gw()
    try
        filename = string("ground_weights.jld")
        ground_weights = load(filename, "ground_weights")

        return ground_weights

    catch e
        #this_size = 50
        ground_weights = rand(Uniform(-2, 1), this_size, this_size)
        filename = string("ground_weights.jld")
        save(filename, "ground_weights", ground_weights)
        return ground_weights
    end
end

function syn(T)
    low = ConstantRate(0.0)
    high = ConstantRate(0.99)
    switch(t; dt = 1) = (t < Int(T / 2)) ? low(t; dt = dt) : high(t; dt = dt)
    the_synapses = QueuedSynapse(Synapse.Alpha())
    syn_current = [switch(t) for t = 1:T]
    excite!(the_synapses, filter(x -> x != 0, syn_current))
    the_synapses
end

function sim_net_darsnack(weight_gain_factor)
    ##
    # Plan network structure stays constant, only synaptic gain varies.
    #
    ##
    ground_weights = get_constant_gw()

    # neuron parameters
    vᵣ = 0
    τᵣ = 1.0
    vth = 1.0
    weights = ground_weights .* weight_gain_factor
    #@show(weights)
    pop_lif = Population(
        weights;
        cell = () -> LIF(τᵣ, vᵣ),
        synapse = Synapse.Alpha,
        threshold = () -> Threshold.Ideal(vth),
    )


    T = 1000

    the_synapses = syn(T)
    ai = [the_synapses for i = 1:this_size]
    spikes = simulate!(pop_lif, T; inputs = ai)
    println("\n simulated: \n")
    rasterplot(spikes) |> display#, label = ["Input 1"])#, "Input 2"])


    return spikes, weights, the_synapses
end



function sim_net_darsnack_learn(weight_gain_factor)
    ##
    # Plan network structure stays constant, only synaptic gain varies.
    #
    ##
    ground_weights = get_constant_gw()

    # neuron parameters
    #vᵣ = 0
    #τᵣ = 1.0
    #vth = 1.0
    weights = ground_weights .* weight_gain_factor
    #@show(weights)
    η₀ = 5.0
    τᵣ = 1.0
    vth = 1.0

    pop = Population(
        weights;
        cell = () -> SRM0(η₀, τᵣ),
        synapse = Synapse.Alpha,
        threshold = () -> Threshold.Ideal(vth),
        learner = STDP(0.5, 0.5, size(weights, 1)),
    )

    # create step input currents
    ai = InputPopulation([ConstantRate(0.8) for i = 1:this_size])
    #ai = [ the_synapses for i in 1:this_size]

    # create network
    net = Network(Dict([:input => ai, :pop => pop]))
    connect!(net, :input, :pop; weights = weights, synapse = Synapse.Alpha)

    # simulate
    #w = Float64[]
    T = 1000

    @time output = simulate!(net, T; dense = true)

    #pop_lif = Population(weights; cell = () -> LIF(τᵣ, vᵣ),
    #                          synapse = Synapse.Alpha,
    #                          threshold = () -> Threshold.Ideal(vth))



    the_synapses = syn(T)
    spikes = output[:pop]
    #spikes = simulate!(pop_lif, T; inputs = ai)
    println("\n simulated: \n")
    rasterplot(spikes) |> display#, label = ["Input 1"])#, "Input 2"])


    return spikes, weights, ai
end

function sim_net_darsnack_used(weight_gain_factor)
    ##
    # Plan network structure stays constant, only synaptic gain varies.
    #
    ##
    ground_weights = get_constant_gw()

    # neuron parameters
    vᵣ = 0
    τᵣ = 1.0
    vth = 1.0
    weights = ground_weights .* weight_gain_factor
    #@show(weights)
    #η₀ = 5.0
    #τᵣ = 1.0
    #vth = 1.0

    pop = Population(
        weights;
        cell = () -> LIF(τᵣ, vᵣ),
        synapse = Synapse.Alpha,
        threshold = () -> Threshold.Ideal(vth),
    )
    # threshold = Threshold.Ideal
    #learner = STDP(0.5, 0.5, size(weights, 1)))

    # create step input currents
    ai = InputPopulation([ConstantRate(0.8) for i = 1:this_size])
    #ai = [ the_synapses for i in 1:this_size]

    # create network
    net = Network(Dict([:input => ai, :pop => pop]))
    connect!(net, :input, :pop; weights = weights, synapse = Synapse.Alpha)

    # simulate
    #w = Float64[]
    T = 1000

    @time output = simulate!(net, T; dense = true)

    #pop_lif = Population(weights; cell = () -> LIF(τᵣ, vᵣ),
    #                          synapse = Synapse.Alpha,
    #                          threshold = () -> Threshold.Ideal(vth))



    the_synapses = syn(T)
    spikes = output[:pop]
    #spikes = simulate!(pop_lif, T; inputs = ai)
    println("\n simulated: \n")
    rasterplot(spikes) |> display#, label = ["Input 1"])#, "Input 2"])

    #clear(pop)
    #clear(net)
    #clear(ai)
    #clear(weights)

    return spikes, weights, ai
end

function get_trains_dars(train_dic::Dict)
    valued = [v for v in values(train_dic)]
    keyed = [k for k in keys(train_dic)]
    cellsa = Array{Union{Missing,Any}}(undef, length(keyed))#, Int(last(findmax(valued)[1])))
    #nac = Int(last(findmax(valued)[1]))

    for (inx, cell_id) in enumerate(1:length(keyed))
        cellsa[inx] = []
    end
    @inbounds for cell_id in keys(train_dic)
        @inbounds for time in train_dic[cell_id]
            append!(cellsa[Int(cell_id)], time * ms)
        end
    end
    #@show(cellsa)
    #cellsa
    cellsb = cellsa[:, 1]
    cellsb
end


#=
function make_net_SNeuralN()
    weights = rand(Uniform(-2,1),25,25)
    pop = Population(weights; cell = () -> LIF(τᵣ, vᵣ),
                              synapse = Synapse.Alpha,
                              threshold = () -> Threshold.Ideal(vth))
    # create input currents
    low = ConstantRate(0.0)
    high = ConstantRate(0.1499)
    switch(t; dt = 1) = (t < Int(T/2)) ? low(t; dt = dt) : high(t; dt = dt)
    n1synapse = QueuedSynapse(Synapse.Alpha())
    n2synapse = QueuedSynapse(Synapse.Alpha())
    excite!(n1synapse, filter(x -> x != 0, [low(t) for t = 1:T]))
    excite!(n2synapse, filter(x -> x != 0, [switch(t) for t = 1:T]))
    voltage_array_size = size(weights)[1]
    voltages = Dict([(i, Float64[]) for i in 1:voltage_array_size])
    cb = () -> begin
        for id in 1:size(pop)
            push!(voltages[id], getvoltage(pop[id]))
        end
    end
    input1 = [ (t; dt) -> 0 for i in 1:voltage_array_size/3]
    input2 = [ n2synapse for i in voltage_array_size/3+1:2*voltage_array_size/3]
    input3 = [ (t; dt) -> 0 for i in 2*voltage_array_size/3:voltage_array_size]
    input = vcat(input2, input1, input3)
    return input,cb,voltages
end
=#
#outputs = simulate!(pop, T; cb = cb, inputs=input)

const Ne = 200;
const Ni = 50
const σee = 1.0
const pee = 0.5
const σei = 1.0
const pei = 0.5

global E
global spkd_ground

function make_net_from_graph_structure(xx)#;

    xx = Int(round(xx))
    @show(xx)
    #h = turan_graph(xx, xx)#, seed=1,cutoff=0.3)

    h = circular_ladder_graph(xx)#, xx)#, seed=1,cutoff=0.3)
    hi = circular_ladder_graph(xx)#, seed=1,cutoff=0.3)
    E = SNN.IZ(; N = Ne, param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -65, d = 8))
    I = SNN.IZ(; N = Ni, param = SNN.IZParameter(; a = 0.1, b = 0.2, c = -65, d = 2))
    #EE = SNN.SpikingSynapse(E, E, :v; σ = σee, p = 1.0)
    EI = SNN.SpikingSynapse(E, I, :v; σ = σei, p = 1.0)
    IE = SNN.SpikingSynapse(I, E, :v; σ = -1.0, p = 1.0)
    II = SNN.SpikingSynapse(I, I, :v; σ = -1.0, p = 1.0)
    # PINningSynapse
    P = [E, I]#, EEA]
    C = [EI, IE, II]#, EEA]
    #EE = SNN.PINningSynapse(E, E, :v; σ=0.5, p=0.8)
    #for n in 1:(N - 1)
    #    SNN.connect!(EE, n, n + 1, 50)
    #end
    #for (i,j) in enumerate(h.fadjlist) println(i,j) end
    EE = SNN.SpikingSynapse(E, E, :v; σ = 0.5, p = 0.8)

    @inbounds for (i, j) in enumerate(h.fadjlist)
        @inbounds for k in j
            SNN.connect!(EE, i, k, 10)
        end
    end

    @inbounds for (i, j) in enumerate(hi.fadjlist)
        @inbounds for k in j
            if i < Ni && k < Ni

                SNN.connect!(EI, i, k, 10)
                SNN.connect!(IE, i, k, 10)
                SNN.connect!(II, i, k, 10)
            end
        end
    end

    #for (i,j) in enumerate(h.fadjlist)
    #    for k in j
    #        SNN.connect!(EI, i, k, 50)
    #    end
    #end
    P = [E, I]#, EEA]
    C = [EE, EI, IE, II]#, EEA]
    return P, C

end


function make_net_SNN(Ne, Ni; σee = 1.0, pee = 0.5, σei = 1.0, pei = 0.5)
    Ne = 200
    Ni = 50

    E = SNN.IZ(; N = Ne, param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -65, d = 8))
    I = SNN.IZ(; N = Ni, param = SNN.IZParameter(; a = 0.1, b = 0.2, c = -65, d = 2))
    EE = SNN.SpikingSynapse(E, E, :v; σ = σee, p = pee)
    EI = SNN.SpikingSynapse(E, I, :v; σ = σei, p = pei)
    IE = SNN.SpikingSynapse(I, E, :v; σ = -1.0, p = 0.5)
    II = SNN.SpikingSynapse(I, I, :v; σ = -1.0, p = 0.5)
    P = [E, I]#, EEA]
    C = [EE, EI, IE, II]#, EEA]
    @show(C)

    return P, C
end
function get_trains(p)
    fire = p.records[:fire]
    x, y = Float32[], Float32[]
    for time in eachindex(fire)
        for neuron_id in findall(fire[time])
            push!(x, time)
            push!(y, neuron_id)
        end
    end
    cellsa = Array{Union{Missing,Any}}(undef, 1, Int(findmax(y)[1]))
    nac = Int(findmax(y)[1])
    for (inx, cell_id) in enumerate(1:nac)
        cellsa[inx] = []
    end
    @inbounds for cell_id in unique(y)
        @inbounds for (time, cell) in collect(zip(x, y))
            if Int(cell_id) == cell
                append!(cellsa[Int(cell_id)], time)

            end

        end
    end

    cellsa

end

#P, C = make_net(Ne, Ni, σee = 0.5, pee = 0.8, σei = 0.5, pei = 0.8, a = 0.02)
#sggcu =[ CuArray(convert(Array{Float32,1},sg)) for sg in spkd_ground ]

#Flux.SGD
#Flux.gpu

function rmse(spkd)
    error = Losses(mean(spkd), spkd; agg = mean)
end

function rmse_depr(spkd)
    total = 0.0
    @inbounds for i = 1:size(spkd, 1)
        total += (spkd[i] - mean(spkd[i]))^2.0
    end
    return sqrt(total / size(spkd, 1))
end


function raster_difference(spkd0, spkd_found)
    maxi0 = size(spkd0)[2]
    maxi1 = size(spkd_found)[2]
    mini = findmin([maxi0, maxi1])[1]
    spkd = ones(mini)
    maxi = findmax([maxi0, maxi1])[1]

    if maxi > 0
        if maxi0 != maxi1
            return sum(ones(maxi))

        end
        if isempty(spkd_found[1, :])
            return sum(ones(maxi))
        end
    end
    spkd = ones(mini)
    @inbounds for i in eachindex(spkd)
        if !isempty(spkd0[i]) && !isempty(spkd_found[i])
            maxt1 = findmax(spkd0[i])[1]
            maxt2 = findmax(spkd_found[i])[1]
            maxt = findmax([maxt1, maxt2])[1]
            if maxt1 > 0.0 && maxt2 > 0.0
                t, S = SpikeSynchrony.SPIKE_distance_profile(
                    unique(sort(spkd0[i])),
                    unique(sort(spkd_found[i]));
                    t0 = 0.0,
                    tf = maxt,
                )
                spkd[i] = SpikeSynchrony.trapezoid_integral(t, S) / (t[end] - t[1])
            end
        end
    end
    spkd
end

#=
using Base.Threads

function raster_difference_threads(spkd0, spkd_found)
    maxi0 = size(spkd0)[2]
    maxi1 = size(spkd_found)[2]
    mini = findmin([maxi0, maxi1])[1]
    spkd = ones(mini)
    maxi = findmax([maxi0, maxi1])[1]
    if maxi > 0
        if maxi0 != maxi1
            return sum(ones(maxi))

        end
        if isempty(spkd_found[1, :])
            return sum(ones(maxi))
        end
    end
    spkd = ones(mini)
    N = length(spkd)
    Threads.foreach(
        Channel(nthreads()) do chnl
            for i in 1:N
                val = 1.0
                if !isempty(spkd0[i]) && !isempty(spkd_found[i])
                    maxt1 = findmax(spkd0[i])[1]
                    maxt2 = findmax(spkd_found[i])[1]
                    maxt = findmax([maxt1, maxt2])[1]
                    if maxt1 > 0.0 && maxt2 > 0.0
                        t, S = SpikeSynchrony.SPIKE_distance_profile(
                            unique(sort(spkd0[i])),
                            unique(sort(spkd_found[i]));
                            t0 = 0.0,
                            tf = maxt,
                        )
                        val = SpikeSynchrony.trapezoid_integral(t, S) / (t[end] - t[1])
                    end
                end
                put!(chnl, val)
                println("thread ", threadid(), ": ", "put ", val)
            end
        end
    end
    spkd
end
=#

#=
function loss(model)
    @show(Ne,Ni)
    @show(model)
    P1, C1 = make_net_SNN(Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 500
    @inbounds for t = 1:sim_length*ms
        E1.I = vec([11.5 for i = 1:sim_length])
        SNN.sim!(P1, C1, 1ms)
    end
    spkd_found = get_trains(P1[1])
    println("Ground Truth \n")
    SNN.raster([E]) |> display
    println("Best Candidate \n")
    SNN.raster([E1]) |> display
    error = raster_difference(spkd_ground, spkd_found)
    error = sum(error)
    @show(error)

    error

end
=#

function loss(model)
    @show(Ne, Ni)
    @show(model)

    σee = model[1]
    pee = model[2]
    σei = model[3]
    pei = model[4]
    P1, C1 = make_net_SNN(Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)
    @show(C1)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 500
    @inbounds for t = 1:sim_length*ms
        E1.I = vec([11.5 for i = 1:sim_length])
        SNN.sim!(P1, C1, 1ms)
    end

    spkd_found = get_trains(P1[1])
    println("Ground Truth \n")
    SNN.raster([E]) |> display
    println("Best Candidate \n")
    SNN.raster([E1]) |> display

    error = raster_difference(spkd_ground, spkd_found)
    error = sum(error)
    @show(error)

    error

end


function eval_best(params)
    #xx = Int(round(params[1]))
    #@show(xx)
    #P1, C1 = make_net_SNN(xx)#,a=a)
    #println("found ",xx)

    σee = params[1]
    pee = params[2]
    σei = params[3]
    pei = params[4]
    P1, C1 = make_net_SNN(Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 500
    @inbounds for t = 1:sim_length*ms
        E1.I = vec([11.5 for i = 1:sim_length])#vec(E_stim[t,:])#[i]#3randn(Ne)
        SNN.sim!(P1, C1, 1ms)
    end

    spkd_found = get_trains(P1[1])
    println("Ground Truth: \n")
    SNN.raster([E]) |> display
    println("candidate: \n")

    SNN.raster([E1]) |> display
    #error = raster_difference(spkd_ground,spkd_found)
    E1, spkd_found

end




#using Evolutionary, MultivariateStats
#range = Any[lower,upper]

# mutation = domainrange(fill(0.5,4))
#
#function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::GA, options)
#    idx = sortperm(state.fitpop)
#record["population"] = population
#    state.fitpop[idx[1:5]]
#    record["fitpop"] = state.fitpop[idx[1:5]]
#end

#using Evolutionary

function Evolutionary.trace!(
    record::Dict{String,Any},
    objfun,
    state,
    population,
    method::GA,
    options,
)
    idx = sortperm(state.fitpop)
    record["fitpop"] = state.fitpop[:]#idx[1:last(idx)]]
    record["pop"] = population[:]
    #record["σ"] = state.
end
#=
function Evolutionary.value!(::Val{:serial}, fitness, objfun, population::AbstractVector{IT}) where {IT}
    @show(typeof(fitness))
    fitness = SharedArrays.SharedArray{Float32}(fitness)
    @time @sync @distributed for i in 1:length(population)

    #pmap(f, [::AbstractWorkerPool], c...; distributed=true, batch_size=) -> collection
        fitness[i] = value(objfun, population[i])
        #println("I'm worker $(myid()), working on i=$i")
    end
    @show(typeof(fitness))

    fitness
    #fitness = Array(fitness)
    #@show(fitness)
end
=#
