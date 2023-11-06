import os.path
import random
from os import path, makedirs
import sys

import numpy as np

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
                       exclude_M1_trajectories=[],
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
        # exclude false trials
        newM1_ids = [x for x in list(range(num_trajectories)) if x not in exclude_M1_trajectories]

        # set random M1 neuron active
        newM1_id = random.choice(newM1_ids)

        M1[layer, newM1_id].baseline = model_params['exc_M1']
        ann.simulate(learning_time)
    else:
        newM1_id = None

    # find most active coordinate
    PMr = np.array(PM.r)
    _, PM_y, PM_x = np.unravel_index(PMr.argmax(), PMr.shape)
    PM_coordinate = state_space[PM_y, PM_x, :]

    # find the most active motor plan
    M1r = np.array(M1.r)
    M1_layer, M1_plan = np.unravel_index(M1r.argmax(), M1r.shape)

    # execute plan
    M1_trajectory = M1_trajectories[M1_plan]
    reached_position = forward_kinematic_arm(thetas=M1_trajectory+init_position,
                                             arm=arm,
                                             return_all_joint_coordinates=False)

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

    return correct, distance, newM1_id


def link_goals_with_bodyrep(id_goal,
                            id_layer,
                            id_output_VA,
                            choose_time=sim_params['max_sim_time'],
                            reward_time=sim_params['reward_time'],
                            SOA_time=sim_params['SOA_time']):

    # create input
    dPFC_baseline = np.zeros(model_params['num_goals'])
    dPFC_baseline[id_goal] = model_params['exc_dPFC']

    S1_baseline = np.zeros(model_params['num_init_positions'])
    S1_baseline[id_layer] = model_params['exc_S1']

    # set baseline from input
    dPFC.baseline = dPFC_baseline
    S1Cortex.baseline = S1_baseline

    # simulate and look up if the "correct" neuron is active
    ann.simulate(choose_time)

    VAr = np.array(VA.r)
    VA_layer, VA_neuron = np.unravel_index(VAr.argmax(), VAr.shape)

    # print(STN_caud.r)
    # print(StrD1_caud.r)
    # print(GPi_caud.r)
    # print(VAr)
    #
    # print(GPe_caud.r)

    if (VA_layer == id_layer) and (VA_neuron == id_output_VA):
        PPTN.baseline = 1.0
        SNc_caud.firing = 1.0
        ann.simulate(reward_time)
        # reset DA
        PPTN.baseline = 0.0
        SNc_caud.firing = 0.0

        correct = 1
    else:
        PPTN.baseline = 0.0
        SNc_caud.firing = 1.0
        ann.simulate(reward_time)
        # reset DA
        PPTN.baseline = 0.0
        SNc_caud.firing = 0.0

        correct = 0

    # SOA
    dPFC.baseline = 0.0
    S1Cortex.baseline = 0.0
    ann.simulate(SOA_time)
    ann.reset(monitors=False, populations=True)

    return correct


def train_body(simID,
               learning_matrix,
               num_layers=model_params['num_init_positions'],
               max_correct=5,
               max_training_trials=50,
               monitoring_training=True):

    # results folder + variables
    save_folder, _ = create_results_folder(simID)

    # monitoring rates and weights
    if monitoring_training:
        pop_monitors = Pop_Monitor(populations=[dPFC, S1Cortex, SNc_caud, StrD1_caud, StrD2_caud, STN_caud,
                                                StrThal_caud, GPe_caud, GPi_caud, VA],
                                   samplingrate=10)

        con_monitors = Con_Monitor([PFCdStrD1_caud, StrD1GPi, PFCdStrD2_caud, StrD2GPe, StrD1SNc_caud])

    print('---------------------------')
    print('Training dorsomedial BG...')
    print('---------------------------\n')

    for i, goal in enumerate(learning_matrix):

        sub_folder = f'Goal[{i}]/'
        if not os.path.exists(save_folder + sub_folder):
            os.mkdir(save_folder + sub_folder)

        error_history = []

        for current_layer in range(num_layers):

            n_trials = 0
            n_correct = 0

            if monitoring_training:
                pop_monitors.start()

            while (n_trials < max_training_trials) & (n_correct < max_correct):
                n_correct += link_goals_with_bodyrep(id_goal=goal[0],
                                                     id_output_VA=goal[1],
                                                     id_layer=current_layer)

                if monitoring_training:
                    con_monitors.extract_weights()

                n_trials += 1

                print(f'Goal: {goal} | Body Position: {current_layer} | Training_trial: {n_trials} | Correct: {n_correct}')

            error_history.append([i, current_layer, n_correct, n_trials])

        # save monitors
        if monitoring_training:
            # Populations
            pop_monitors.stop()
            pop_monitors.save(save_folder + sub_folder)

            # Weights
            con_monitors.save_cons(save_folder + sub_folder)
            con_monitors.reset()

        # save error history
        np.savetxt(save_folder + sub_folder + 'training_history_dm.txt', np.array(error_history))


def train_motor_network(simID,
                        state_space,
                        possible_trajectories,
                        VA_amp=model_params['exc_VA'] ,
                        num_goals=model_params['num_goals'],
                        num_layers=model_params['num_init_positions'],
                        max_training_trials=50,
                        max_correct=2,
                        monitoring_training=True):

    # results folder + variables
    save_folder, _ = create_results_folder(simID)

    # monitoring variables
    if monitoring_training:
        pop_monitors = Pop_Monitor([VA, PM, latStrD1, SNr, VL, M1, latSNc], samplingrate=10)
        con_monitors = Con_Monitor([StrD1SNc_put,] + [Connection for Connection in PMStrD1_putamen])

    print('---------------------------')
    print('Training dorsolateral BG...')
    print('---------------------------\n')

    for current_layer in range(num_layers):

        print(f'\nLayer: {current_layer}')
        sub_folder = f'Layer[{current_layer}]/'
        if not os.path.exists(save_folder + sub_folder):
            os.mkdir(save_folder + sub_folder)

        error_history = []

        if monitoring_training:
            pop_monitors.start()

        for goal in range(num_goals):

            VA_input = np.zeros(num_goals)
            VA_input[goal] = VA_amp

            n_trials = 0
            n_correct = 0
            false_trajectories = []

            while (n_trials < max_training_trials) & (n_correct < max_correct):
                correct, error_distance, M1_id = learn_motor_skills(layer=current_layer,
                                                                    thal_input_array=VA_input,
                                                                    M1_trajectories=possible_trajectories[current_layer],
                                                                    exclude_M1_trajectories=false_trajectories,
                                                                    state_space=state_space)

                if monitoring_training:
                    con_monitors.extract_weights()

                if not correct and M1_id is not None:
                    false_trajectories.append(M1_id)

                error_history.append((goal, error_distance))
                n_correct += correct
                n_trials += 1

                print(f'Goal: {goal} | Training_trial: {n_trials} | Correct: {n_correct}')

        # save monitors
        if monitoring_training:
            # Populations
            pop_monitors.stop()
            pop_monitors.save(save_folder + sub_folder)

            # Weights
            con_monitors.save_cons(save_folder + sub_folder)
            con_monitors.reset()

        # save error history
        np.savetxt(save_folder + sub_folder + 'training_history_dl.txt', np.array(error_history))


def test_network(simID,
                 test_matrix,
                 state_space,
                 possible_trajectories,
                 choose_time=sim_params['max_sim_time'],
                 SOA_time=sim_params['SOA_time'],
                 monitoring_test=True):

    # results folder + variables
    _, save_folder = create_results_folder(simID)

    # monitoring variables
    if monitoring_test:
        pop_monitors = Pop_Monitor(populations=[VA, PM, latStrD1, SNr, VL, M1, latSNc] + [dPFC, S1Cortex, SNc_caud,
                                                StrD1_caud, StrD2_caud, StrThal_caud, GPe_caud, GPi_caud],
                                   samplingrate=10)

        pop_monitors.start()

    n_correct = 0
    for test_set in test_matrix:

        id_goal, id_layer = test_set
        init_position = np.radians(model_params['moving_arm_positions'][id_layer])
        arm = model_params['moving_arm']

        # create input
        dPFC_baseline = np.zeros(model_params['num_goals'])
        dPFC_baseline[id_goal] = model_params['exc_dPFC']

        S1_baseline = np.zeros(model_params['num_init_positions'])
        S1_baseline[id_layer] = model_params['exc_S1']

        # set baseline from input
        dPFC.baseline = dPFC_baseline
        S1Cortex.baseline = S1_baseline

        # simulate
        ann.simulate_until(max_duration=choose_time, population=M1)

        # find most active coordinate
        PMr = np.array(PM.r)
        _, PM_y, PM_x = np.unravel_index(PMr.argmax(), PMr.shape)
        PM_coordinate = state_space[PM_y, PM_x, :]

        # find the most active motor plan
        M1r = np.array(M1.r)
        M1_layer, M1_plan = np.unravel_index(M1r.argmax(), M1r.shape)

        # execute plan
        M1_trajectory = possible_trajectories[M1_layer][M1_plan]
        reached_position = forward_kinematic_arm(thetas=M1_trajectory + init_position,
                                                 arm=arm,
                                                 return_all_joint_coordinates=False)

        # Does the selected trajectory reach the most active PM coordinate?
        distance = np.linalg.norm(PM_coordinate - reached_position)
        n_correct += distance < 5.0  # in [mm]

        # SOA
        dPFC.baseline = 0.0
        S1Cortex.baseline = 0.0
        ann.simulate(SOA_time)
        ann.reset(monitors=False, populations=True)

    # save monitors
    if monitoring_test:
        # Populations
        pop_monitors.stop()
        pop_monitors.save(save_folder)


def run_full_network(simID, monitors_training=True, monitors_test=True):
    # add connections between the dorsomedial and dorsolateral network
    add_connections()

    # create state space + trajectories
    state_space = create_state_space()
    possible_trajectories = create_trajectories()

    # compiling network
    if not path.exists(compile_folder):
        makedirs(compile_folder)
    ann.compile(directory=compile_folder + f'annarchy_motorBG[{simID}]')

    # run training dorsomedial network
    # train_body(simID,
    #            learning_matrix=model_params['training_set'],
    #            monitoring_training=monitors_training)

    # run training dorsolateral network
    train_motor_network(simID,
                        state_space=state_space,
                        possible_trajectories=possible_trajectories,
                        monitoring_training=monitors_training)

    # test network
    test_network(simID,
                 test_matrix=model_params['test_set'],
                 possible_trajectories=possible_trajectories,
                 state_space=state_space,
                 monitoring_test=monitors_test)


if __name__ == '__main__':

    simID = sys.argv[1]
    run_full_network(simID, monitors_training=False, monitors_test=False)
