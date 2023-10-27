import numpy as np
from parameters import model_params, state_space_limits


# sigmoid isn't perfectly symmetrical around its turning point
def normal_space(start, stop, num):
    def sigmoid(num):
        values = np.linspace(0, 20, num-1)
        res = 1/(1+np.exp(-values+5))
        return res

    offset = abs(start) + stop

    x = sigmoid(num)

    # uniform linspace
    y = np.linspace(0, offset, num-1)

    return np.append(x * y + start, stop)


def sin_space(start, stop, num):
    offset = abs(start) + stop

    x = np.sin(np.linspace(0, np.pi/2, num, endpoint=True))

    # uniform linspace
    y = np.linspace(0, offset, num, endpoint=True)

    return x * y + start


def gauss_wrapper(func, sigma=50.0):
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if isinstance(res, np.ndarray | np.generic):
            return np.exp(-0.5 * (res * res) / (sigma * sigma))
        elif isinstance(res, tuple | list):
            res = np.array(res)
            return np.exp(-0.5 * (res * res) / (sigma * sigma))
        else:
            return None
    return wrapper


def goal_coordinates(arm=model_params['resting_arm'],
                     resting_arm_angles=model_params['resting_arm_positions'],
                     rel_forearm_tactile_points=model_params['rel_position_goal']):

    from forward_kinematic import return_tactile_point

    coordinates_pos = np.zeros((len(resting_arm_angles),
                                len(rel_forearm_tactile_points),
                                2))

    for i, angles in enumerate(resting_arm_angles):
        for j, rel_goal in enumerate(rel_forearm_tactile_points):
            coordinates_pos[i, j, :] = return_tactile_point(theta=angles,
                                                            arm=arm,
                                                            percentile=rel_goal,
                                                            radians=False)

    return coordinates_pos


def create_trajectories(cov, num=model_params['num_trajectories'], fix_seed=False):
    from forward_kinematic import return_tactile_point
    from inverse_kinematic import inverse_kinematic_gradient_decent
    from parameters import model_params

    if fix_seed:
        np.random.seed(2)
    else:
        np.random.seed()

    goals = goal_coordinates()
    length = np.random.random()
    width = np.random.normal(0, 20)

    random_point = return_tactile_point(theta=model_params['moving_arm_positions'][0],
                                        arm=model_params['moving_arm'],
                                        percentile=length,
                                        delta_shoulder=width,
                                        radians=False)



def min_space_between_goals():
    """ This function returns the minimal distance between different possible goal positions. The minimal value
        should than be used to adjust the step size in creating the state space."""

    from itertools import combinations

    goal_coord = goal_coordinates()

    # flatten array to just coordinates
    goal_coord = goal_coord.reshape((goal_coord.shape[0] * goal_coord.shape[1], 2))

    # get every combination of coordinates without duplicates
    x_combinations = np.array(list(combinations(np.unique(goal_coord[:, 0]), 2)))
    y_combinations = np.array(list(combinations(np.unique(goal_coord[:, 1]), 2)))

    x_distances = np.abs(x_combinations[:, 0] - x_combinations[:, 1])
    y_distances = np.abs(y_combinations[:, 0] - y_combinations[:, 1])

    return np.min(x_distances), np.min(y_distances)


def create_state_space(x_bound=state_space_limits['x_boundaries'],
                       y_bound=state_space_limits['y_boundaries'],
                       resolution_factor=1.0):

    x_lowerbound, x_upperbound = x_bound
    y_lowerbound, y_upperbound = y_bound

    step_size_x, step_size_y = min_space_between_goals()

    x = np.arange(start=x_lowerbound, stop=x_upperbound+step_size_x, step=np.ceil(resolution_factor * step_size_x))
    y = np.arange(start=y_lowerbound, stop=y_upperbound+step_size_y, step=np.ceil(resolution_factor * step_size_y))

    xx, yy = np.meshgrid(x, y)
    xy = np.dstack((xx, yy))

    return xy


def return_goal_indeces():
    goal_coord = goal_coordinates()
    state_space = create_state_space()

    num_angles, num_tactile_points, _ = goal_coord.shape

    goal_indeces = np.zeros((num_angles, num_tactile_points, 2))
    for i_angle in range(num_angles):
        for j_goal in range(num_tactile_points):
            # find min in x
            goal_indeces[i_angle, j_goal, 0] = np.argmin(np.abs(state_space[0, :, 0] - goal_coord[i_angle, j_goal, 0]))
            # find min in y
            goal_indeces[i_angle, j_goal, 1] = np.argmin(np.abs(state_space[:, 0, 1] - goal_coord[i_angle, j_goal, 1]))

    return goal_indeces.astype(int)


def bivariate_gauss(mu_index, sigma, norm=False, plot=False, limit=None):
    from scipy.stats import multivariate_normal

    xy = create_state_space()
    i, j = mu_index
    mu = xy[i, j, :]

    rv = multivariate_normal(mu, cov=sigma * np.identity(2))
    a = rv.pdf(xy)

    if norm:
        a /= np.max(a)

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = ax.contourf(a)
        plt.colorbar(img)
        plt.show()

    if limit is not None:
        a[a < limit] = None

    return a


@gauss_wrapper
def generate_weights(thetas, arm, rad=True):
    # generate end effectors
    from forward_kinematic import forward_kinematic_arm
    points = []
    for joint_angle in thetas:
        points.append(forward_kinematic_arm(thetas=joint_angle,
                                            arm=arm,
                                            radians=rad,
                                            return_all_joint_coordinates=False))

    distance = np.zeros((len(points), len(points)))
    for i, point_fix in enumerate(points):
        for j, point_alt in enumerate(points):
            distance[i, j] = np.linalg.norm(point_fix - point_alt)

    return distance


if __name__ == '__main__':
    print(create_state_space().shape)

    # bivariate_gauss([5, 8], 20.0, plot=True)
    create_trajectories(10, 10)
