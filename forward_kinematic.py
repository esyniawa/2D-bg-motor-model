import numpy as np

from parameters import joint_lengths


# Calculate the Transformation Matrix A according to the DH convention
def DH_matrix(a, d, alpha, theta, radians=True):
    if not radians:
        alpha = np.radians(alpha)
        theta = np.radians(theta)

    A = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                  [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                  [0, np.sin(alpha), np.cos(alpha), d],
                  [0, 0, 0, 1]])

    return A


def forward_kinematic_arm(thetas, arm, radians=True, return_all_joint_coordinates=False, delta_shoulder=0):
    if isinstance(thetas, tuple | list):
        thetas = np.array(thetas)

    if not radians:
        thetas = np.radians(thetas)

    if arm == 'right':
        const = 1
    elif arm == 'left':
        const = - 1
        thetas = np.array((np.pi, 0)) - thetas
    else:
        raise ValueError('Please specify if the arm is right or left!')

    A0 = DH_matrix(const * joint_lengths['a0'], 0, 0, 0)
    A1 = DH_matrix(joint_lengths['a1']+delta_shoulder, 0, 0, thetas[0])
    A2 = DH_matrix(joint_lengths['a2'], 0, 0, thetas[1])

    # Shoulder -> elbow
    A01 = A0 @ A1
    # Elbow -> hand
    A12 = A01 @ A2

    if return_all_joint_coordinates:
        return np.column_stack(([0, 0], A0[:2, 3], A01[:2, 3], A12[:2, 3]))
    else:
        return A12[:2, 3]


def return_tactile_point(theta, arm, percentile,
                         modality=2, radians=True, delta_shoulder=0):

    if modality not in [0, 1, 2]:
        raise ValueError('Define: 0 = head -> shoulder; 1 = shoulder -> elbow; 2 = elbow -> hand')

    if not radians:
        theta = np.radians(theta)

    coordinates_arm = forward_kinematic_arm(theta,
                                            arm=arm,
                                            return_all_joint_coordinates=True,
                                            delta_shoulder=delta_shoulder)

    start = coordinates_arm[:, modality]
    end = coordinates_arm[:, modality+1]

    coordinate_point = start + percentile * (end - start)

    return coordinate_point


def construct_arms(theta_right, theta_left, radians=True, do_plot=True):
    if not radians:
        theta_right = np.radians(theta_right)
        theta_left = np.radians(theta_left)

    coordinates_right = forward_kinematic_arm(theta_right, arm='right', return_all_joint_coordinates=True)
    coordinates_left = forward_kinematic_arm(theta_left, arm='left', return_all_joint_coordinates=True)

    if not do_plot:
        return coordinates_right, coordinates_left
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(coordinates_right[0, :], coordinates_right[1, :], 'b')
        ax.plot(coordinates_left[0, :], coordinates_left[1, :], 'g')

        plt.ylabel('y in [mm]', fontsize=10)
        plt.xlabel('x in [mm]', fontsize=10)

        plt.show()


if __name__ == '__main__':
    print(return_tactile_point([20, 90], 'left', 0.1, radians=False))

    a = construct_arms(theta_right=[20, 90], theta_left=[20, 90], radians=False, do_plot=True)
