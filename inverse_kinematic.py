import numpy as np

from parameters import joint_lengths
from forward_kinematic import forward_kinematic_arm


# returns the jacobian matrix for a 2D system with two variable joints
def jacobian(thetas, arm):
    if isinstance(thetas, tuple | list):
        thetas = np.array(thetas)

    if arm == 'right':
        theta1, theta2 = thetas
        const = 1
    elif arm == 'left':
        theta1 = np.pi - thetas[0]
        theta2 = - thetas[1]
        const = - 1
    else:
        raise ValueError('Please specify if the arm is right one or left one!')

    # Calculate the derivatives of the end effector position with respect to the joint angles
    J11 = -joint_lengths['a1'] * np.sin(theta1) - joint_lengths['a2']*np.sin(theta1 + theta2)
    J12 = -joint_lengths['a2'] * np.sin(theta1 + theta2)
    J21 = joint_lengths['a1'] * np.cos(theta1) + joint_lengths['a2'] * np.cos(theta1 + theta2)
    J22 = joint_lengths['a2'] * np.cos(theta1 + theta2)

    # Assemble the Jacobian matrix
    J = const * np.array([[J11, J12], [J21, J22]])

    return J


# returns the movement trajectory from initial joint angles to an end effector in a 2D plane
def inverse_kinematic_gradient_decent(end_effector, starting_angles, arm,
                                      max_iterations=500,
                                      learning_rate=0.1,
                                      abort_criteria=1, # in [mm]
                                      radians=True,
                                      do_plot=False):

    if isinstance(end_effector, tuple | list):
        end_effector = np.array(end_effector)

    if not radians:
        starting_angles = np.radians(starting_angles)

    thetas = np.array(starting_angles)
    trajectory = []
    for i in range(max_iterations):
        # Compute the forward kinematics for the current joint angles
        current_position = forward_kinematic_arm(thetas, arm)
        trajectory.append(current_position)

        # Calculate the error between the current end effector position and the desired end point
        error = end_effector - current_position

        # abort when error is smaller than the breaking condition
        if np.linalg.norm(error) < abort_criteria:
            break

        # Calculate the Jacobian matrix for the current joint angles
        J = jacobian(thetas, arm)

        delta_thetas = learning_rate * np.linalg.inv(J) @ error
        thetas += delta_thetas

    if not do_plot:
        return thetas, np.array(trajectory)
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        former_position = forward_kinematic_arm(starting_angles, arm, return_all_joint_coordinates=True)
        end_position = forward_kinematic_arm(thetas, arm, return_all_joint_coordinates=True)
        trajectory = np.array(trajectory)

        ax.plot(former_position[0, :], former_position[1, :], 'b--')
        ax.plot(end_position[0, :], end_position[1, :], 'b')

        ax.scatter(trajectory[:, 0], trajectory[:, 1])

        plt.ylabel('y in [mm]', fontsize=10)
        plt.xlabel('x in [mm]', fontsize=10)

        plt.show()


def touch_joint(starting_joint_angles,
                resting_joint_angles,
                resting_arm, # modality
                joint_to_touch, # 0 = head -> shoulder; 1 = shoulder -> elbow; 2 = elbow -> hand
                percentile=0.5, # ratio
                radians=True,
                do_plot=False,
                delta_shoulder=0):

    if not joint_to_touch in [0, 1, 2]:
        raise ValueError('Define: 0 = head -> shoulder; 1 = shoulder -> elbow; 2 = elbow -> hand')

    if not radians:
        starting_joint_angles = np.radians(starting_joint_angles)
        resting_joint_angles = np.radians(resting_joint_angles)

    if resting_arm == 'right':
        moving_arm = 'left'
    elif resting_arm == 'left':
        moving_arm = 'right'

    coordinates_rest_arm = forward_kinematic_arm(resting_joint_angles, resting_arm,
                                                 return_all_joint_coordinates=True,
                                                 delta_shoulder=delta_shoulder)
    start = coordinates_rest_arm[:, joint_to_touch]
    end = coordinates_rest_arm[:, joint_to_touch+1]

    touch_point = start + percentile * (end - start)

    # calculate movement trajectory to the touch point
    new_thetas, trajectory = inverse_kinematic_gradient_decent(
        end_effector=touch_point,
        starting_angles=starting_joint_angles,
        arm=moving_arm
    )

    if not do_plot:
        return new_thetas, trajectory
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        initial_position_moving_arm = forward_kinematic_arm(starting_joint_angles, moving_arm, return_all_joint_coordinates=True)
        last_position_moving_arm = forward_kinematic_arm(new_thetas, moving_arm, return_all_joint_coordinates=True)

        ax.plot(coordinates_rest_arm[0, :], coordinates_rest_arm[1, :], 'g')
        ax.plot(initial_position_moving_arm[0, :], initial_position_moving_arm[1, :], 'b--')
        ax.plot(last_position_moving_arm[0, :], last_position_moving_arm[1, :], 'b')

        ax.scatter(trajectory[:, 0], trajectory[:, 1], marker='+')
        ax.scatter(touch_point[0], touch_point[1], c='r')

        plt.ylabel('y in [mm]', fontsize=10)
        plt.xlabel('x in [mm]', fontsize=10)

        plt.show()


def check_relation_statespace_jointangles(starting_joint_angles,
                                          resting_joint_angles,
                                          resting_arm,
                                          joint_to_touch, # 0 = head -> shoulder; 1 = shoulder -> elbow; 2 = elbow -> hand
                                          num,
                                          radians=True,
                                          do_plot=False,
                                          delta_shoulder=0):

    x_space = np.linspace(0,1, num)

    joint_angles = []
    trajectories = []
    for x in x_space:
        angle, trajectory = touch_joint(starting_joint_angles=starting_joint_angles,
                                        resting_joint_angles=resting_joint_angles,
                                        resting_arm=resting_arm, # modality
                                        joint_to_touch=joint_to_touch, # 0 = head -> shoulder; 1 = shoulder -> elbow; 2 = elbow -> hand
                                        percentile=x, # ratio
                                        radians=radians,
                                        do_plot=False,
                                        delta_shoulder=delta_shoulder)


        joint_angles.append(angle)
        trajectories.append(trajectory)

    joint_angles = np.array(joint_angles)
    diff_angles = joint_angles[1:, :] - joint_angles[:-1, :]

    import matplotlib.pyplot as plt

    if resting_arm == 'right':
        moving_arm = 'left'
    elif resting_arm == 'left':
        moving_arm = 'right'

    initial_position_moving_arm = forward_kinematic_arm(starting_joint_angles, moving_arm, radians=radians, return_all_joint_coordinates=True)
    position_resting_arm = forward_kinematic_arm(resting_joint_angles, resting_arm, radians=radians, return_all_joint_coordinates=True)

    fig = plt.figure("Test", figsize=(20, 15))

    ax = fig.add_subplot(2, 1, 1)

    plt.ylabel("y [mm]")
    plt.xlabel("x [mm]")

    for trajectory in trajectories:
        ax.scatter(trajectory[:, 0], trajectory[:, 1], marker='+', c='b', s=1)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='o', c='r', s=50)

    ax.plot(position_resting_arm[0, :], position_resting_arm[1, :], 'g', linewidth = 3.0)
    ax.plot(initial_position_moving_arm[0, :], initial_position_moving_arm[1, :], 'b', linewidth = 3.0)

    ax = fig.add_subplot(2, 2, 3)
    x = np.arange(num)
    ax.plot(x, joint_angles[:, 0], 'r', label='Shoulder angle', linewidth=3.0)
    ax.plot(x, joint_angles[:, 1], 'y', label='Elbow angle', linewidth=3.0)

    plt.ylabel("angle(i) [rad]")
    plt.xlabel("i")

    y = np.linspace(0.5, 2, 11)

    plt.xticks(x)
    plt.yticks(y)
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    x = np.arange(num-1)
    y = np.linspace(-0.05, 0.05, 11)
    ax.plot(x, diff_angles[:, 0], 'r', label='Shoulder angle differences', linewidth = 3.0)
    ax.plot(x, diff_angles[:, 1], 'y', label='Elbow angle differences', linewidth = 3.0)

    plt.ylabel("angle(i+1) - angle(i) [rad]")
    plt.xlabel("i")

    max_shoulder = np.max(diff_angles[:, 0])
    min_shoulder = np.min(diff_angles[:, 0])

    max_elbow = np.max(diff_angles[:, 1])
    min_elbow = np.min(diff_angles[:, 1])

    y = np.linspace(np.min((min_elbow, min_shoulder)), np.max((max_shoulder, max_elbow)), 11)
    plt.xticks(x)
    plt.yticks(y)
    ax.legend()

    plt.show()


def inverse_kinematic_bads(end_effector, starting_angles, arm, radians=True):
    from pybads.bads import BADS
    from parameters import joint_limits

    if not radians:
        starting_angles = np.radians(starting_angles)

    def error_function(thetas, end_point=end_effector, moving_arm=arm):
        current_point = forward_kinematic_arm(thetas, moving_arm)

        error = np.linalg.norm(end_point - current_point)
        return error

    target = error_function

    bads = BADS(target, starting_angles,
                plausible_lower_bounds=joint_limits['lower_thetas'],
                plausible_upper_bounds=joint_limits['upper_thetas'])

    optimize_results = bads.optimize()
    new_angles = optimize_results['x']

    if radians:
        return new_angles
    else:
        return np.degrees(new_angles)


def check_limits(thetas, radians=True):
    from parameters import joint_limits

    if isinstance(thetas, tuple | list):
        thetas = np.array(thetas)

    if not radians:
        thetas = np.radians(thetas)

    ret = all(thetas >= joint_limits['lower_thetas']) & all(thetas <= joint_limits['upper_thetas'])

    return ret


if __name__ == '__main__':
    test_inverse_kinematic = False
    test_self_touch = False
    test_bads = False
    test_distance = True

    if test_inverse_kinematic:
        a = inverse_kinematic_gradient_decent(end_effector=[-100, 150],
                                              starting_angles=[0, 90],
                                              arm='right',
                                              radians=False,
                                              do_plot=True)

    if test_self_touch:
        touch_joint(
            starting_joint_angles=[20, 110],
            resting_joint_angles=[20, 110],
            resting_arm='right',
            joint_to_touch=2,
            percentile=0.25,
            radians=False,
            do_plot=True
        )

    if test_bads:
        print(inverse_kinematic_bads(end_effector=np.array([-100, 150]),
                                     starting_angles=[20, 110],
                                     arm='left',
                                     radians=False))

    if test_distance:
        distance(starting_joint_angles=[20, 110],
                 resting_joint_angles=[90, 90],
                 resting_arm='left',
                 joint_to_touch=2,
                 num=3,
                 radians=False,
                 do_plot=True,
                 delta_shoulder=0)
