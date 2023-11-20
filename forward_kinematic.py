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
    import matplotlib.pyplot as plt
    from parameters import model_params, state_space_limits
    from inverse_kinematic import inverse_kinematic_gradient_decent
    from functions import bivariate_gauss, return_tactile_indeces, tactile_coordinates
    import os

    from colour import Color

    nthetas = len(model_params['resting_arm_positions'])
    arm_id = 2
    resting_theta = model_params['resting_arm_positions'][arm_id]
    moving_theta = model_params['resting_arm_positions'][arm_id]
    goal = tactile_coordinates()[2, 3]

    min_x, max_x = state_space_limits['x_boundaries']

    n = 40
    min_shoulder, min_elbow = model_params['resting_arm_positions'][0]
    max_shoulder, max_elbow = model_params['resting_arm_positions'][-1]

    shoulder_theta = np.linspace(min_shoulder,
                                 max_shoulder, n)

    elbow_theta = np.linspace(min_elbow,
                              max_elbow, n)

    thetas = np.dstack((shoulder_theta, elbow_theta)).reshape((n, 2))

    dist_colors = list(Color("#ffffff").range_to(Color("#cc0000"), int(n/2)))

    dist_colors = dist_colors + [Color("#cc0000")] + dist_colors[::-1]

    green = Color("#2eb82e")
    colors = list(green.range_to(Color("#006600"), nthetas))

    fig, ax = plt.subplots()
    for i, theta_left in enumerate(model_params['resting_arm_positions']):

        coor_right, coor_left = construct_arms(model_params['moving_arm_positions'][i], theta_left,
                                               radians=False, do_plot=False)

        if i == arm_id:
            tact_points = [return_tactile_point(theta_left, 'left', percentile=per, radians=False) for per in model_params['rel_position_goal']]
            for tact_point in tact_points:
                ax.scatter(tact_point[0], tact_point[1], c='black', s=40, zorder=2)

            ax.plot(coor_right[0, :], coor_right[1, :], color=colors[i].hex, linewidth=5.0, zorder=1, alpha=0.4)
            ax.plot(coor_left[0, :], coor_left[1, :], color=colors[i].hex, linewidth=5.0, zorder=1)

        else:
            ax.plot(coor_right[0, :], coor_right[1, :], color=colors[i].hex, linestyle='dashed', zorder=0, alpha=0.4)
            ax.plot(coor_left[0, :], coor_left[1, :], color=colors[i].hex, linestyle='dashed', zorder=0)

    for c, theta in enumerate(thetas):
        point = forward_kinematic_arm(theta, 'left', radians=False, return_all_joint_coordinates=False)
        ax.scatter(point[0], point[1], zorder=0, c=dist_colors[c].hex, s=10)

    ax.set_xlim(state_space_limits['x_boundaries'])
    ax.set_ylim(state_space_limits['y_boundaries'])

    plt.ylabel('y in [mm]', fontsize=16)
    plt.xlabel('x in [mm]', fontsize=16)

    folder = 'figures/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.savefig(folder + 'positions.png')

    plt.show()
    plt.close()

    # Plot touch point and visual fixation
    fig, ax = plt.subplots()

    y, x = return_tactile_indeces()[arm_id, 3]
    map1 = bivariate_gauss([x, y], 50, norm=True)
    map1[map1 < 0.01] = np.NaN

    map2 = 2*bivariate_gauss([x-5, y-5], 100, norm=True)
    map2[map2 < 0.01] = np.NaN

    coor_right, coor_left = construct_arms(moving_theta, resting_theta, radians=False, do_plot=False)

    ax.plot(coor_right[0, :], coor_right[1, :], color=colors[i].hex, linewidth=5.0, zorder=2, alpha=0.3)
    ax.plot(coor_left[0, :], coor_left[1, :], color=colors[i].hex, linewidth=5.0, zorder=2, alpha=0.3)

    ax.set_xlim(state_space_limits['x_boundaries'])
    ax.set_ylim(state_space_limits['y_boundaries'])
    plt.ylabel('y in [mm]', fontsize=16)
    plt.xlabel('x in [mm]', fontsize=16)

    ax.contourf(map1, cmap='Blues', origin='lower', zorder=2, vmin=0.0, vmax=1.0,
                extent=state_space_limits['x_boundaries'] + state_space_limits['y_boundaries'],
                alpha=0.4)

    ax.contourf(map2, cmap='Reds', origin='lower', zorder=1, vmin=0.0, vmax=1.0,
                extent=state_space_limits['x_boundaries'] + state_space_limits['y_boundaries'],
                alpha=0.6)

    plt.savefig(folder + 'premotor.png')
    plt.show()
    plt.close()

    # motor plan
    fig, ax = plt.subplots()

    ax.plot(coor_right[0, :], coor_right[1, :], color=colors[arm_id].hex, linewidth=3.0, zorder=1, alpha=0.3, linestyle='dashed')
    ax.plot(coor_left[0, :], coor_left[1, :], color=colors[arm_id].hex, linewidth=5.0, zorder=1, alpha=0.3)

    new_theta, trajectory = inverse_kinematic_gradient_decent(goal, starting_angles=moving_theta, arm='right', radians=False)
    new_position = forward_kinematic_arm(new_theta, 'right', radians=True, return_all_joint_coordinates=True)

    ax.plot(new_position[0, :], new_position[1, :], color=colors[arm_id].hex, linewidth=5.0, zorder=2)
    ax.scatter(trajectory[:, 0], trajectory[:, 1], zorder=0, c='b', s=10)

    ax.set_xlim(state_space_limits['x_boundaries'])
    ax.set_ylim(state_space_limits['y_boundaries'])
    plt.ylabel('y in [mm]', fontsize=16)
    plt.xlabel('x in [mm]', fontsize=16)

    ax.contourf(map2, cmap='Reds', origin='lower', zorder=2, vmin=0.0, vmax=1.0,
                extent=state_space_limits['x_boundaries'] + state_space_limits['y_boundaries'],
                alpha=0.5)

    plt.savefig(folder + 'movement_plan.png')
    plt.show()
    plt.close()
