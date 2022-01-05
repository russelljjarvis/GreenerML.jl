#using ClearStacktrace
using Plots
using JLD
using Test
using Revise
using SpikingNN
using Revise
unicodeplots()

const net_size = 50
#struct SquareWave{T<:Real} <: AbstractInput
#    τ::T
#end

#SpikingNN.evaluate!(input:: SquareWave, t::Integer; dt::Real = 1.0) =
#    (mod(t, ceil(Int, input.τ / dt)) * dt < input.τ / 2) ? t : zero(t)
#(input:: SquareWave)(t::Integer; dt::Real = 1.0) = SpikingNN.evaluate!(input, t; dt = dt)
#SpikingNN.evaluate!(inputs::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<: SquareWave} =
#    ifelse.(mod.(t, ceil.(Int, inputs.τ ./ dt)) .* dt .< inputs.τ ./ 2, t, zero(t))
#sw = SpikingNN.SquareWave(200)
#=
struct SquareWave{T<:Real} <: AbstractInput
    τ::T
    amp::T
end
sw = SquareWave(200,100)
@show(sw)
@show(sw.τ)

function SpikingNN.evaluate!(input:: SquareWave, t::Integer; dt::Real = 1.0)
    (mod(t, ceil(Int, input.τ / dt)) * dt < input.τ / 2) ? t : zero(t)(input:: SquareWave)(t::Integer; dt::Real = 1.0)
    SpikingNN.evaluate!(input, t; dt = dt)
end

function SpikingNN.evaluate!(inputs::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<: SquareWave}
    ifelse.(mod.(t, ceil.(Int, inputs.τ ./ dt)) .* dt .< inputs.τ ./ 2, t, zero(t))
end
=#
function get_constant_gw()
    ##
    # Load a connectome from disk.
    ##
    try
        filename = string("ground_weights.jld")
        ground_weights = load(filename,"ground_weights")
        return ground_weights

    catch e
        ground_weights = rand(Uniform(-2,1),net_size,net_size)
        filename = string("ground_weights.jld")
        save(filename, "ground_weights", ground_weights)
        return ground_weights
    end
end

function sim_net_darsnack_learn(weight_gain_factor)
    ground_weights = get_constant_gw()
    weights = ground_weights .* weight_gain_factor
    η₀ = 5.0
    τᵣ = 1.0
    vth = 1.0

    pop = Population(weights; cell = () -> SRM0(η₀, τᵣ),
                              synapse = Synapse.Alpha,
                              threshold = () -> Threshold.Ideal(vth),
                              learner = STDP(0.5, 0.5, size(weights, 1)))

    # create step input currents
    #sw = StepCurrent(10)
    sw = StepCurrent(10)#*10.0
    ai = InputPopulation([sw for i in 1:net_size])

    # create network
    net = Network(Dict([:input => ai, :pop => pop]))
    connect!(net, :input, :pop; weights = weights, synapse = Synapse.Alpha)
    T = 1000
    output = simulate!(net, T; dense = true)
    #the_synapses = syn(T)
    spikes = output[:pop]
    println("\n simulated: \n")
    rasterplot(spikes)|>display
    return spikes,weights,ai
end

function sim_net_darsnack_used(weight_gain_factor)
    ground_weights = get_constant_gw()
    # neuron parameters
    vᵣ = 0
    τᵣ = 1.0
    vth = 1.0
    weights = ground_weights .* weight_gain_factor

    net = Population(weights; cell = () ->LIF(τᵣ, vᵣ),
                              synapse = Synapse.Alpha,
                              threshold = () -> Threshold.Ideal(vth))
                              # threshold = Threshold.Ideal
                              #learner = STDP(0.5, 0.5, size(weights, 1)))
    η₀ = 5.0
    τᵣ = 1.0
    vth = 1.0
    net2 = Population(weights; cell = () -> SRM0(η₀, τᵣ),
                            synapse = Synapse.Alpha,
                            threshold = () -> Threshold.Ideal(vth),
                            learner = STDP(0.5, 0.5, size(weights, 1)))

    # create step input currents

    #sw = SpikingNN.SquareWave(2000)
    #sw = ConstantRate(0.9)


    #pop = Population(connectivity_matrix, neurons; ϵ = Synapse.Alpha)
    # setclass(pop, 1, :input)
    # setclass(pop, 2, :input)

    # create input currents
    #low = ConstantRate(0.1)
    #high = ConstantRate(0.99)
    switch(t; dt) = (t < Int(T/2)) ? zeros(t) : 10*ones(t)

    # excite neurons
    #excite!(pop[1], low, T; response = Synapse.Alpha())
    T = 1000

    excite!(net2[:], switch, T; response = Synapse.Alpha())
    #excite!(pop[1], i, T; response = Synapse.Alpha())

    #sw = StepCurrent(0.10)#*10.0
    #@show(sw)
    #ai = InputPopulation([sw for i in 1:net_size])
    #@show(ai)
    #ai = InputPopulation([SquareWave(0.8) for i in 1:net_size])
    # create network
    #net = Network(Dict([:input => ai, :pop => pop]))
    connect!(net, :pop; weights = weights, synapse = Synapse.Alpha)
    # here
    output = simulate!(net, T; dense = true)
    #
    spikes = output[:pop]
    println("\n simulated: \n")
    rasterplot(spikes)|>display
    return spikes,weights,ai
end

##
# Ground truth
##

const weight_gain_factor = 0.77

spkd_ground_dic1,weights1,synpases1 = sim_net_darsnack_used(weight_gain_factor)
nspkd_ground_dic2,weights2,synpases2 = sim_net_darsnack_used(weight_gain_factor)
@test spkd_ground_dic1==nspkd_ground_dic2


spkd_ground_dic1,weights1,synpases1 = sim_net_darsnack_learn(weight_gain_factor)
nspkd_ground_dic2,weights2,synpases2 = sim_net_darsnack_learn(weight_gain_factor)
@test spkd_ground_dic1==nspkd_ground_dic2
