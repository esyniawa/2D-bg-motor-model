import ANNarchy as ann
from os import path, makedirs
import sys

from dorsomedial_network import *
from dorsolateral_network import *

from projections import dmThal_PM_connection
from parameters import model_params, sim_params, create_results_folder, compile_folder

from forward_kinematic import forward_kinematic_arm

from functions import create_state_space, create_trajectories
from monitoring import Pop_Monitor, Con_Monitor


def add_connections():
    # adds the connection between both networks
    VAPM_con = ann.Projection(pre=VA, post=PM, target='exc')
    weights = dmThal_PM_connection(sigma=10)
    VAPM_con.connect_from_matrix(weights)


def learn_motor_skills(layer,
                       thal_input_array,
                       M1_trajectories,
                       state_space,
                       choose_time=sim_params['max_sim_time'],
                       learning_time=sim_params['learning_time'],
                       reward_time=sim_params['reward_time'],
                       SOA_time=sim_params['SOA_time']):

    num_trajectories, _ = M1_trajectories.shape
    init_position = np.radians(model_params['moving_arm_positions'][layer])
    arm = model_params['moving_arm']

    VA[layer, :].baseline = thal_input_array
    ann.simulate_until(max_duration=choose_time, population=M1)

    # Find max and execute motorplan
    M1r_max = np.max(M1.r)

    if M1r_max < model_params['threshold_M1r']:
        newM1_baseline = np.zeros(num_trajectories)
        newM1_baseline[np.random.randint(num_trajectories)] = model_params['M1_amp']

        M1[layer, :].baseline = newM1_baseline
        ann.simulate(learning_time)

    # find most active coordinate
    PMr = np.array(PM.r)
    _, PM_y, PM_x = np.unravel_index(PMr.argmax(), PMr.shape)
    PM_coordinate = state_space[PM_y, PM_x, :]

    print(PM_coordinate)

    # find the most active motor plan
    M1r = np.array(M1.r)
    M1_layer, M1_plan = np.unravel_index(M1r.argmax(), M1r.shape)

    # execute plan
    M1_trajectory = M1_trajectories[M1_plan]
    reached_position = forward_kinematic_arm(thetas=M1_trajectory+init_position,
                                             arm=arm,
                                             return_all_joint_coordinates=False)

    print(reached_position)

    # Does the selected trajectory reach the most active PM coordinate?
    distance = np.linalg.norm(PM_coordinate - reached_position)
    correct = distance < 5.0 # in [mm]

    # if yes, release DA
    if correct:
        reward.baseline = 1.0
        latSNc.firing = 1.0
        ann.simulate(reward_time)
        # reset DA
        latSNc.firing = 0.0
        reward.baseline = 0.0

    VA.baseline = 0.0
    M1.baseline = 0.0
    ann.simulate(SOA_time)
    ann.reset(monitors=False, populations=True)

    return correct, distance


def train_motor_network(simID,
                        VA_amp=2.0,
                        num_goals=model_params['num_goals'],
                        num_layers=model_params['num_init_positions'],
                        max_training_trials=100,
                        max_correct=2,
                        monitoring_training=True):

    # results folder + variables
    save_folder = create_results_folder(simID)

    # monitoring variables
    if monitoring_training:
        pop_monitors = Pop_Monitor([VA, PM, latStrD1, SNr, VA, M1, latSNc], samplingrate=10)
        con_monitors = Con_Monitor([StrD1SNc_put,] + [Connection for Connection in CortexStrD1_putamen])

    print('Training BG...\n')
    print('--------------\n')

    # define state space and possible trajectories
    state_space = create_state_space()
    possible_trajectories = create_trajectories()

    for current_layer in range(num_layers):

        print(f'Layer: {current_layer}\n')
        sub_folder = f'Layer[{current_layer}]/'

        error_history = []

        if monitoring_training:
            pop_monitors.start()

        for goal in range(num_goals):

            VA_input = np.zeros(num_goals)
            VA_input[goal] = VA_amp

            n_trials = 0
            n_correct = 0

            while (n_trials < max_training_trials) & (n_correct < max_correct):
                correct, error_distance = learn_motor_skills(layer=current_layer,
                                                             thal_input_array=VA_input,
                                                             M1_trajectories=possible_trajectories[current_layer],
                                                             state_space=state_space)

                if monitoring_training:
                    con_monitors.extract_weights()

                error_history.append((goal, error_distance))
                n_correct += correct
                n_trials += 1

                print(f'Goal: {goal} | Training_trial: {n_trials}. | Correct: {n_correct}')

        # save monitors
        if monitoring_training:
            # Populations
            pop_monitors.stop()
            pop_monitors.save(save_folder + sub_folder)

            # Weights
            con_monitors.save_cons(save_folder + sub_folder)
            con_monitors.reset()

        # save error history
        np.savetxt(save_folder + sub_folder + 'error_history.txt', np.array(error_history))


def run_full_network():
    pass


if __name__ == '__main__':

    simID = sys.argv[1]

    # add connections between the dorsomedial and dorsolateral network
    add_connections()

    # compiling network
    if not path.exists(compile_folder):
        makedirs(compile_folder)
    ann.compile(directory=compile_folder + f'annarchy_motorBG[{simID}]')

    # run training dorsolateral network
    train_motor_network(simID)
