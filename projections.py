import ANNarchy as ann
import itertools

def rangeX(iterations):
    """
    Multidimensional iterator using all possible combinations within the given
    boundaries e.g. rangeX((2,3)) => 00 01 02 10 11 12
    """
    if not isinstance(iterations,  tuple | list):
        raise AttributeError('Input should be a tuple or a list!')
    return itertools.product(*map(range, iterations))

def STN_GPi_connection(preDim, postDim, weight=1.0):
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
