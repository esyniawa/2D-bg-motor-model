import ANNarchy as ann

from parameters import model_params
from model_definitions import *

from projections import laterals_layerwise
from functions import create_state_space
y_dim, x_dim, _ = create_state_space().shape

dl_baseline_DA = 0.1

#################################################
############# Populations #######################

PM = ann.Population(name='PM', geometry=(model_params['num_init_positions'], y_dim, x_dim), neuron=LinearNeuron)
latStrD1 = ann.Population(name='dorsolateral_StrD1', geometry=model_params['dim_lateral_BG'], neuron=LinearNeuron)

SNr = ann.Population(name='SNr', geometry=model_params['dim_lateral_BG'], neuron=LinearNeuron)
SNr.tau = 5.0
SNr.noise = 0.05
SNr.baseline = 1.1

VL = ann.Population(name='VL', geometry=model_params['dim_lateral_BG'], neuron=LinearNeuron)
VL.tau = 8.0
VL.noise = 0.05
VL.baseline = 0.9

M1 = ann.Population(name='M1', geometry=model_params['dim_lateral_BG'], neuron=LinearNeuron, stop_condition="r > 2.0")
M1.tau = 20.0
M1.noise = 0.01
M1.baseline = 0.0

latSNc = ann.Population(name='SNc_caud', geometry=1, neuron=DopamineNeuron)
reward = ann.Population(name='Excitatory reward population', geometry=1, neuron=LinearNeuron)
reward_inh = ann.Population(name='Inhibitory reward population', geometry=1, neuron=LinearNeuron)

###############################################
####### Projections ###########################

# Laterals
SNrSNr_putamen = ann.Projection(pre=SNr, post=SNr, target='exc', synapse=ReversedSynapse)
SNr_laterals = laterals_layerwise(preDim=model_params['dim_lateral_BG'], postDim=model_params['dim_lateral_BG'], weight=0.1)
SNrSNr_putamen.connect_from_matrix(SNr_laterals)
SNrSNr_putamen.reversal = 1.2


VLVL_putamen = ann.Projection(pre=VL, post=VL, target='inh')
VL_laterals = laterals_layerwise(preDim=model_params['dim_lateral_BG'], postDim=model_params['dim_lateral_BG'], weight=0.4)
VLVL_putamen.connect_from_matrix(VL_laterals)


# FF TODO: This connection should be an ANNarchy object not a list
PMStrD1_putamen = []
for init_pos in range(model_params['num_init_positions']):
    PMStrD1_putamen.append(ann.Projection(pre=PM[init_pos, :, :],
                                              post=latStrD1[init_pos, :],
                                              target='exc',
                                              synapse=DAPostCovarianceNoThreshold,
                                              name=f'PMStrD1_putamen_{init_pos}'))

    PMStrD1_putamen[init_pos].connect_all_to_all(weights=0.0)
    PMStrD1_putamen[init_pos].tau = 2200
    PMStrD1_putamen[init_pos].regularization_threshold = 1.2
    PMStrD1_putamen[init_pos].tau_alpha = 20.0
    PMStrD1_putamen[init_pos].baseline_dopa = 0.1
    PMStrD1_putamen[init_pos].K_dip = 0.1
    PMStrD1_putamen[init_pos].K_burst = 1.2
    PMStrD1_putamen[init_pos].threshold_post = 0.1
    PMStrD1_putamen[init_pos].threshold_pre = 0.0
    PMStrD1_putamen[init_pos].DA_type = 1

StrD1SNr = ann.Projection(pre=latStrD1, post=SNr, target='inh')
StrD1SNr.connect_one_to_one(weights=0.8)

SNrVA_putamen = ann.Projection(pre=SNr, post=VL, target='inh')
SNrVA_putamen.connect_one_to_one(weights=0.6) #0.5 #0.5

VLM1_putamen = ann.Projection(pre=VL, post=M1, target='exc')
VLM1_putamen.connect_one_to_one(weights=1.0) #1.0 #3.0

SNcStrD1_put = ann.Projection(pre=latSNc, post=latStrD1, target='dopa')
SNcStrD1_put.connect_all_to_all(weights=1.0)

SNcSNr_put = ann.Projection(pre=latSNc, post=SNr, target='dopa')
SNcSNr_put.connect_all_to_all(weights=1.0)

HvSNc = ann.Projection(pre=reward, post=latSNc, target='exc')
HvSNc.connect_all_to_all(weights=1.0)
HvSNc_inh = ann.Projection(pre=reward_inh, post=latSNc, target='inh')
HvSNc_inh.connect_all_to_all(weights=1.0)

StrD1SNc_put = ann.Projection(pre=latStrD1, post=latSNc, target='inh', synapse=DAPrediction, name='StrD1SNc_put')
StrD1SNc_put.connect_all_to_all(weights=0.0)
StrD1SNc_put.tau = 10000
StrD1SNc_put.baseline_dopa = dl_baseline_DA

# FB
M1Str = ann.Projection(pre=M1, post=latStrD1, target='exc')
M1Str.connect_one_to_one(weights=0.5)
