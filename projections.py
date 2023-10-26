def STN_GPi_connection(preDim, postDim, weight=1.0):
    '''
    Returns a connection matrix usable in ANNarchy.
    The idea behind this connection is, that every pre-synaptic neuron i connects with every post-synaptic
    neuron j which do not map the features as neuron i:
    if i == j: w(ij) = None; else: w = [weight]
    '''
    import numpy as np

    if not isinstance(postDim,  tuple | list):
        raise AttributeError

    i, j = postDim
    w = np.array([[[None]*i]*j]*preDim)

    for pre_i in range(preDim):
        for post_j in range(i):
            if pre_i != post_j:
                w[pre_i, :, post_j] = weight

    w = w.T
    # ANNarchy needs the weight matrix in the shape (i*j, preDim)
    return w.reshape((i*j), preDim)
