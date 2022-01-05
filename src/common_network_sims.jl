
function get_constant_gw()
    try
        filename = string("ground_weights.jld")
        ground_weights = load(filename, "ground_weights")

        return ground_weights

    catch e
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

    weights = ground_weights .* weight_gain_factor
    η₀ = 5.0
    τᵣ = 1.0 # membrane time constant
    vth = 1.0 # Voltage threshold (must cross to fire).

    pop = Population(
        weights;
        cell = () -> SRM0(η₀, τᵣ),
        synapse = Synapse.Alpha,
        threshold = () -> Threshold.Ideal(vth),
        learner = STDP(0.5, 0.5, size(weights, 1)),
    )

    # create step input currents
    # artificial input
    ai = InputPopulation([ConstantRate(0.8) for i = 1:this_size])

    # create network
    net = Network(Dict([:input => ai, :pop => pop]))
    connect!(net, :input, :pop; weights = weights, synapse = Synapse.Alpha)
    T = 1000
    @time output = simulate!(net, T; dense = true)
    the_synapses = syn(T)
    spikes = output[:pop]
    println("\n simulated: \n")
    rasterplot(spikes) |> display
    return spikes, weights, ai
end

function sim_net_darsnack_used(weight_gain_factor)
    """
    A simulation where network structure (connectome) stays constant.
    Only synaptic gain varies.
    """
    ground_weights = get_constant_gw()

    # neuron parameters
    vᵣ = 0
    τᵣ = 1.0
    vth = 1.0
    weights = ground_weights .* weight_gain_factor
    pop = Population(
        weights;
        cell = () -> LIF(τᵣ, vᵣ),
        synapse = Synapse.Alpha,
        threshold = () -> Threshold.Ideal(vth),
    )

    # create step input currents
    # artificial input (ai)
    ai = InputPopulation([ConstantRate(0.8) for i = 1:this_size])
    # create network
    net = Network(Dict([:input => ai, :pop => pop]))
    connect!(net, :input, :pop; weights = weights, synapse = Synapse.Alpha)

    # simulate
    T = 1000
    @time output = simulate!(net, T; dense = true)
    the_synapses = syn(T)
    spikes = output[:pop]
    println("\n simulated: \n")
    rasterplot(spikes) |> display
    return spikes, weights, ai
end




function get_trains_dars(train_dic::Dict)
    """
    Get spike trains from a Darsnak SpikingNN backend simulation.
    """
    valued = [v for v in values(train_dic)]
    keyed = [k for k in keys(train_dic)]
    cellsa = Array{Union{Missing,Any}}(undef, length(keyed))
    for (inx, cell_id) in enumerate(1:length(keyed))
        cellsa[inx] = []
    end
    @inbounds for cell_id in keys(train_dic)
        @inbounds for time in train_dic[cell_id]
            append!(cellsa[Int(cell_id)], time * ms)
        end
    end
    cellsa[:, 1]
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
