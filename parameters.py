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

#############################################
######### Network parameters ################

# setup params
model_params = {
    'nGoals': 4,
    'nBodySchema': 5,
    'nMovements': 20,
}

# dorsomedial
model_params['dim_Str'] = (5, 5)
model_params['dim_BG'] = (model_params['nGoals'], model_params['nBodySchema'])

#############################################
###### Simulation parameters ################

sim_params = {
    'nThreads': 2
}