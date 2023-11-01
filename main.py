import ANNarchy as ann

from dorsomedial_network import *
from dorsolateral_network import *

from projections import dmThal_PM_connection
from parameters import model_params, sim_params

from forward_kinematic import forward_kinematic_arm

from functions import create_state_space, create_trajectories
from monitoring import Pop_Monitor, Con_Monitor


def add_connections():
    # TODO: Add connections from the dorsomedial connection

    # dorsolateral
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
    M1r = np.array(M1.r)
    M1r_max = np.max(M1r)

    if M1r_max < model_params['threshold_M1r']:
        newM1_baseline = np.zeros(num_trajectories)
        newM1_baseline[np.random.randint(num_trajectories)] = model_params['M1_amp']

        M1[layer, :].baseline = newM1_baseline
        ann.simulate(learning_time)

    PM_argmax = np.argmax(PM.r)
    PM_coordinate = state_space[PM_argmax[0], PM_argmax[1], :]

    M1_argmax = np.argmax(M1.r)
    M1_trajectory = M1_trajectories[M1_argmax]
    reached_position = forward_kinematic_arm(thetas=M1_trajectory+init_position,
                                             arm=arm)

    if np.linalg.norm(PM_coordinate - reached_position) < 1.0:
        reward.baseline = 1.0
        latSNc.firing = 1.0
        ann.simulate(reward_time)
        latSNc.firing = 0.0
        reward.baseline = 0.0

    VA.baseline = 0.0
    M1.baseline = 0.0
    ann.simulate(SOA_time)
    ann.reset(monitors=False, populations=True)

def train_motor_network():

    state_space = create_state_space()
    possible_trajectories = create_trajectories()

    pass


def run_full_network():
    pass



