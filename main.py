import ANNarchy as ann

from dorsomedial_network import *
from dorsolateral_network import *

from projections import dmThal_PM_connection
from parameters import model_params, sim_params

from functions import goal_coordinates
from monitoring import Pop_Monitor, Con_Monitor


def add_connections():
    VAPM_con = ann.Projection(pre=VA, post=PM, target='exc')
    weights = dmThal_PM_connection(sigma=10)
    VAPM_con.connect_from_matrix(weights)


def train_motor_network(choose_time=sim_params['max_sim_time'],
                        learning_time=sim_params['learning_time'],
                        reward_time=sim_params['reward_time'],
                        SOA_time=sim_params['SOA_time']):
    pass


def run_full_network():
    pass



