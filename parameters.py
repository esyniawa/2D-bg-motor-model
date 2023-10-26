import numpy as np
import os

results_folder = 'results/'
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

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
model_params['rel_position_goal'] = (0.25, 0.5, 0.75)
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
model_params['dim_mStr'] = (5, 5)
model_params['dim_BG'] = (model_params['num_goals'], model_params['num_init_positions'])

# dorsolateral
model_params['num_trajectories'] = 16

#############################################
###### Simulation parameters ################

sim_params = {
    'num_threads': 2
}
