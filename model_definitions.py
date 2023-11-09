import ANNarchy as ann

# Neuron definitions
BaselineNeuron = ann.Neuron(
    parameters="""
        tau_up = 1.0 : population
        tau_down = 20.0 : population
        baseline = 0.0
        noise = 0.0 : population
    """,
    equations="""
        base = baseline + noise * Uniform(-1.0,1.0): min=0.0
        dr/dt = if (baseline>0.01): (base-r)/tau_up else: -r/tau_down : min=0.0
    """,
    name = "Baseline Neuron",
    description = "Neuron with baseline to be set. "
)

LinearNeuron = ann.Neuron(
    parameters="""
        tau = 10.0
        baseline = 0.0
        noise = 0.0
        lesion = 1.0
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
        r = lesion*pos(mp)
    """
)

LinearNeuron_trace = ann.Neuron(
    parameters="""
        tau = 10.0
        baseline = 0.0
        noise = 0.0
        tau_trace = 120.0
        lesion = 1.0
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
        r = lesion*pos(mp)
        tau_trace*dtrace/dt + trace = r
    """
)

DopamineNeuron = ann.Neuron(
    parameters="""
        tau = 10.0
        firing = 0
        inhibition = 0.0
        baseline = 0.0
        exc_threshold = 0.0
        factor_inh = 10.0
    """,
    equations="""
        ex_in = if (sum(exc)>exc_threshold): 1 else: 0
        s_inh = sum(inh)
        aux =   if (firing>0): 
                    (ex_in)*(pos(1.0-baseline-s_inh) + baseline) + (1-ex_in)*(-factor_inh*sum(inh))  
                else: baseline
        tau*dmp/dt + mp =  aux
        r = if (mp>0.0): mp else: 0.0
    """
)

# Synapse definitions
ReversedSynapse = ann.Synapse(
    parameters="""
        reversal = 1.2
    """,
    psp="""
        w*pos(reversal-pre.r)
    """,
    name="Reversed Synapse",
    description="Higher pre-synaptic activity lowers the synaptic transmission and vice versa."
)

# DA_typ = 1  ==> D1 type  DA_typ = -1 ==> D2 type
DAPostCovarianceNoThreshold = ann.Synapse(
    parameters="""
        tau=1000.0
        tau_alpha=10.0
        regularization_threshold=1.0
        baseline_dopa = 0.1
        K_burst = 1.0
        K_dip = 0.4
        DA_type = 1
        threshold_pre=0.0
        threshold_post=0.0
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha = pos(post.mp - regularization_threshold)
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(post.r -  mean(post.r) - threshold_post) * (pre.r - mean(pre.r) - threshold_pre)
	    condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0
        dopa_mod =  if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum
                    else: condition_0*DA_type*K_dip*dopa_sum
        delta = (dopa_mod* trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post))
        tau*dw/dt = delta : min=0
    """
)

# Inhibitory synapses STRD1 -> SNr and STRD2 -> GPe
DAPreCovariance_inhibitory = ann.Synapse(
    parameters="""
    tau=1000.0
    tau_alpha=10.0
    regularization_threshold=1.0
    baseline_dopa = 0.1
    K_burst = 1.0
    K_dip = 0.4
    DA_type= 1
    threshold_pre=0.0
    threshold_post=0.0
    negterm = 1
    """,
    equations="""
        tau_alpha*dalpha/dt = pos( -post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.r) - post.r  - threshold_post)
        aux = if (trace>0): negterm else: 0
        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum
        delta = dopa_mod * trace - alpha * pos(trace)
        tau*dw/dt = delta : min=0
    """
)


DAPrediction = ann.Synapse(
    parameters="""
        tau = 100000.0
        baseline_dopa = 0.1
   """,
   equations="""
       aux = if (post.sum(exc)>0): 1.0 else: 3.0
       delta = aux*pos(post.r - baseline_dopa)*pos(pre.r - mean(pre.r))
       tau*dw/dt = delta : min=0
   """
)
