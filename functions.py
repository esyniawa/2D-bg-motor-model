import numpy as np

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


def bivariate_gauss(max_position, num_cell_per_ax, velocity):
    from scipy.stats import multivariate_normal

    x, y = np.mgrid[-max_position:max_position:(2 * max_position / num_cell_per_ax),
                    -max_position:max_position:(2 * max_position / num_cell_per_ax)]

    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rr = multivariate_normal(velocity,
                             cov=np.identity(2))
    a = rr.pdf(pos)
    return a


if __name__ == '__main__':

    myspace = normal_space(-10, 20, 20)
    print(myspace)