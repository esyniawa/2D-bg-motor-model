import numpy as np
import ANNarchy as ann
from functions import rangeX


def STN_GPi_connection(preDim, postDim, weight=1.0):
    """
    Returns a connection matrix usable in ANNarchy.
    The idea behind this connection is, that every pre-synaptic neuron i connects with every post-synaptic
    neuron j which do not map the features as neuron i. The connection rule reads like this:
    if i == j: w(ij) = None; else: w = [weight]
    """

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
    return w.reshape(i*j, preDim)


def S1_STN_connection():

    from functions import generate_weights
    from parameters import model_params

    w = generate_weights(thetas=model_params['moving_arm_positions'],
                         arm=model_params['moving_arm'])

    return w.T


def dmThal_PM_connection(sigma, limit=None):
    """
    Returns a connection matrix usable in ANNarchy.
    This custom connection maps the different goals to limb positions in the state space. The certainty of the position
    is modulated by a bivariate gaussian. The variance can be passed to the function with the parameter sigma.
    """
    from functions import bivariate_gauss, create_state_space, return_goal_indeces

    goals = return_goal_indeces()
    dim_body_maps, dim_thal, _ = goals.shape
    y_dim, x_dim, _ = create_state_space().shape

    # I wish that could be prettier
    w = np.empty((dim_thal, dim_body_maps, x_dim, y_dim, dim_body_maps))
    for i, init_position in enumerate(goals):
        for j, goal in enumerate(init_position):
            w[j, i, :, :, i] = bivariate_gauss(mu_index=goal, sigma=sigma, norm=True, limit=limit).T

    w = w.T
    return w.reshape(y_dim * x_dim * dim_body_maps, dim_thal * dim_body_maps)


# TODO: implement this function as the connector between PM and Striatum D1 in the dorsolateral network
def dim_to_one(pre, post, preDim, weight=1.0, delays=0):
    """
    Creates a connection that connects all neurons along the specified
    pre-dimension (preDim) to one neuron of the pre-population.
    """
    assert post.geometry[0] == pre.geometry[preDim]


    synapses = ann.CSR()
    nGeometry = pre.geometry[:preDim] + pre.geometry[preDim + 1:]
    for n in range(post.geometry[0]):
        pre_ranks = []
        for m in rangeX(nGeometry):
            mn = m[:preDim] + (n,) + m[preDim + 1:]
            pre_ranks.append(pre.rank_from_coordinates(mn))
        synapses.add(n, pre_ranks, [weight], [delays])
    return synapses
