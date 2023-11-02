import numpy as np
import os


def create_results_folder(simID):
    results_folder = f'results/train_motor_network[{simID}]/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder


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
    'y_boundaries': (0, 200)
}

#############################################
######### Network parameters ################
# setup params
model_params = {}

# relative positions of the goal on the forearm
model_params['rel_position_goal'] = (0.25, 0.5, 0.75, 1.0)   # must be <= 1.0
model_params['num_goals'] = len(model_params['rel_position_goal'])

# moving arm parameters
model_params['moving_arm'] = 'right'
# joint angles in [°]
model_params['moving_arm_positions'] = ((0, 90),
                                        (10, 100),
                                        (20, 110))

# resting arm positions
model_params['resting_arm'] = 'left'
# joint angles in [°] (the left arm will be touched)
model_params['resting_arm_positions'] = ((0, 90),
                                         (10, 100),
                                         (20, 110))

# number of initial positions
model_params['num_init_positions'] = len(model_params['moving_arm_positions'])

# dorsomedial
model_params['dim_medial_Str'] = (5, 5)
model_params['dim_medial_BG'] = (model_params['num_init_positions'], model_params['num_goals'])

# dorsolateral
model_params['num_trajectories'] = 16

model_params['dim_lateral_BG'] = (model_params['num_init_positions'], model_params['num_trajectories'])
# Threshold that M1 has to exceed to disable learning via motor babbling
model_params['threshold_M1r'] = 0.1
model_params['M1_amp'] = 1.0

model_params['min_distance_possible_positions'] = 10.0 # in [mm]
model_params['sample_sigma'] = 50.0 # in [mm]

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
