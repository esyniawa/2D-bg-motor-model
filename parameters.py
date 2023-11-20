import numpy as np
import os


def create_results_folder(simID):
    train_folder = f'results/BG_motor_network[{simID}]/training/'
    test_folder = f'results/BG_motor_network[{simID}]/testing/'

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    return train_folder, test_folder


compile_folder = 'networks/'
if not os.path.exists(compile_folder):
    os.mkdir(compile_folder)

#############################################
##### DH parameters + constraints ###########

joint_lengths = {
    'a0': 50, # in [mm]
    'a1': 220, # in [mm]
    'a2': 160  # in [mm]
}

joint_limits = {
    'lower_thetas': np.radians([-20, 0]),
    'upper_thetas': np.radians([110, 170])
}

state_space_limits = {
    'x_boundaries': (-300, 300),
    'y_boundaries': (-10, 210)
}

#############################################
######### Network parameters ################
# setup params
model_params = {}

# relative positions of the goal on the forearm
model_params['num_goals'] = 1
model_params['rel_position_goal'] = np.arange(start=0, stop=1.01, step=0.2)   # must be <= 1.0
model_params['num_forearm_points'] = len(model_params['rel_position_goal'])

# resting arm positions
model_params['resting_arm'] = 'left'
# joint angles in [°] (the left arm will be touched)
model_params['resting_arm_positions'] = ((0, 90),
                                         (10, 100),
                                         (20, 110),
                                         (30, 120),
                                         (40, 130))

# moving arm parameters
model_params['moving_arm'] = 'right'
# joint angles in [°]
model_params['moving_arm_positions'] = model_params['resting_arm_positions'][::-1]

# number of initial positions
model_params['num_init_positions'] = len(model_params['moving_arm_positions'])

# training set for the dorsomedial loop [id_goal_input, id_goal_output (in VA)]
model_params['training_set'] = ((0, 0),)


if len(model_params['training_set']) > model_params['num_goals']:
    raise AttributeError

# test set for testing the 2 loops [id_goal, id_body_position]
model_params['test_set'] = ((0, 0),
                            (0, 1),
                            (0, 2))

# dorsomedial
model_params['dim_medial_Str'] = (5, 5)
model_params['dim_medial_BG'] = (model_params['num_init_positions'], model_params['num_forearm_points'])

model_params['exc_dPFC'] = 1.0
model_params['exc_S1'] = 1.0

model_params['sd_distance_RBF_S1STN'] = 100 # in [mm]

# dorsolateral
model_params['num_trajectories'] = 16

# projection between VA and PM
model_params['bivariate_gauss_sigma'] = 15
model_params['dim_lateral_BG'] = (model_params['num_init_positions'], model_params['num_trajectories'])
# Threshold that M1 has to exceed to disable learning via motor babbling
model_params['threshold_M1r'] = 0.2

model_params['exc_M1'] = 1.0
model_params['exc_VA'] = 3.0

model_params['min_distance_possible_positions'] = 10.0 # in [mm]
model_params['sample_sigma'] = 10.0 # in [mm]

#############################################
###### Simulation parameters ################

sim_params = {
    'num_threads': 2,
    # Presentation time parameters
    'max_sim_time': 1000, # [ms]
    'reward_time': 100, # [ms]
    'SOA_time': 100, # [ms]
    'learning_time': 300 # [ms]
}
