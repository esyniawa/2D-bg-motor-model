import ANNarchy as ann
import numpy as np

from model_definitions import *
from parameters import model_params
from projections import STN_GPi_connection, laterals_layerwise, S1_STN_connection

baseline_dopa_caud = 0.1

''' Basal Ganglia network to integrate a goal depending on the body schema'''

# Input populations
dPFC = ann.Population(name="dPFC", geometry=model_params['num_goals'], neuron=BaselineNeuron)

S1Cortex = ann.Population(name="S1",
                          geometry=model_params['num_init_positions'],
                          neuron=BaselineNeuron)

PPTN = ann.Population(name="PPTN", geometry=1, neuron=BaselineNeuron)

# Populations of the dorsomedial network
StrD1_caud = ann.Population(name="StrD1_caud", geometry=model_params['dim_medial_Str'], neuron=LinearNeuron_trace)
StrD1_caud.noise = 0.01
StrD1_caud.lesion = 1.0

StrD2_caud = ann.Population(name="StrD2_caud", geometry=model_params['dim_medial_Str'], neuron=LinearNeuron_trace)
StrD2_caud.noise = 0.01
StrD2_caud.lesion = 1.0

STN_caud = ann.Population(name="STN_caud", geometry=model_params['num_init_positions'], neuron=LinearNeuron_trace)
STN_caud.noise = 0.01
STN_caud.lesion = 1.0

GPi_caud = ann.Population(name="GPi_caud", geometry=model_params['dim_medial_BG'], neuron=LinearNeuron)
GPi_caud.noise = 0.01
GPi_caud.baseline = 1.9

GPe_caud = ann.Population(name="GPe_caud", geometry=model_params['dim_medial_BG'], neuron=LinearNeuron)
GPe_caud.noise = 0.05
GPe_caud.baseline = 1.0

VA = ann.Population(name="VA", geometry=model_params['dim_medial_BG'], neuron=LinearNeuron)
VA.noise = 0.025
VA.baseline = 1.0

StrThal_caud = ann.Population(name="StrThal_caud", geometry=model_params['dim_medial_BG'], neuron=LinearNeuron)
StrThal_caud.noise = 0.01
StrThal_caud.baseline = 0.4

SNc_caud = ann.Population(name='SNc_caud', geometry=1, neuron=DopamineNeuron)
SNc_caud.exc_threshold = 0.2
SNc_caud.baseline = baseline_dopa_caud
SNc_caud.factor_inh = 10.0

# Projections
# TODO: thresholds must be set to the input accordingly
# direct pathway
PFCdStrD1_caud = ann.Projection(pre=dPFC, post=StrD1_caud, target='exc', synapse=DAPostCovarianceNoThreshold, name='PFCdStrD1_caud')
PFCdStrD1_caud.connect_all_to_all(weights=ann.Normal(0.5, 0.2))
PFCdStrD1_caud.tau = 100.0 #100
PFCdStrD1_caud.regularization_threshold = 2.0
PFCdStrD1_caud.tau_alpha = 10.0
PFCdStrD1_caud.baseline_dopa = baseline_dopa_caud
PFCdStrD1_caud.K_dip = 0.5
PFCdStrD1_caud.K_burst = 1.0
PFCdStrD1_caud.DA_type = 1
PFCdStrD1_caud.threshold_pre = 0.0
PFCdStrD1_caud.threshold_post = 0.2 # init 0.0

StrD1GPi = ann.Projection(pre=StrD1_caud, post=GPi_caud, target='inh', synapse=DAPreCovariance_inhibitory, name='StrD1GPi_caud')
StrD1GPi.connect_all_to_all(weights=ann.Normal(0.5, 0.1)) # scale by numer of stimuli #var 0.01
StrD1GPi.tau = 500.0 #700 #550
StrD1GPi.regularization_threshold = 2.25 #1.5
StrD1GPi.tau_alpha = 5.0 # 20.0
StrD1GPi.baseline_dopa = baseline_dopa_caud
StrD1GPi.K_dip = 0.9
StrD1GPi.K_burst = 1.0
StrD1GPi.threshold_post = 0.0 #0.1 #0.3
StrD1GPi.threshold_pre = 0.1 # 0.15
StrD1GPi.DA_type = 1

# indirect pathway
PFCdStrD2_caud = ann.Projection(pre=dPFC, post=StrD2_caud, target='exc', synapse=DAPostCovarianceNoThreshold, name='PFCdStrD2_caud')
PFCdStrD2_caud.connect_all_to_all(weights=ann.Normal(0.4, 0.2)) #0.005
PFCdStrD2_caud.tau = 50.0
PFCdStrD2_caud.regularization_threshold = 1.0
PFCdStrD2_caud.tau_alpha = 20.0
PFCdStrD2_caud.baseline_dopa = baseline_dopa_caud
PFCdStrD2_caud.K_dip = 0.2
PFCdStrD2_caud.K_burst = 2.0 # init 1.0
PFCdStrD2_caud.DA_type = -1
PFCdStrD2_caud.threshold_pre = 0.0
PFCdStrD2_caud.threshold_post = 0.0

StrD2GPe = ann.Projection(pre=StrD2_caud, post=GPe_caud, target='inh', synapse=DAPreCovariance_inhibitory, name='StrD2GPe_caud')
StrD2GPe.connect_all_to_all(weights=0.0)
StrD2GPe.tau = 50.0
StrD2GPe.regularization_threshold = 1.5
StrD2GPe.tau_alpha = 20.0
StrD2GPe.baseline_dopa = baseline_dopa_caud
StrD2GPe.K_dip = 0.2
StrD2GPe.K_burst = 2.0
StrD2GPe.threshold_post = 0.0
StrD2GPe.threshold_pre = 0.1
StrD2GPe.DA_type = -1

GPeGPi = ann.Projection(pre=GPe_caud, post=GPi_caud, target='inh', name='GPeGPi_caud')
GPeGPi.connect_one_to_one(weights=1.0)

# hyperdirect pathway
S1STN_caud = ann.Projection(pre=S1Cortex, post=STN_caud, target='exc', name='S1STN_caud')
S1STN_caud.connect_one_to_one(weights=0.3)

# S1STN_weights = S1_STN_connection(weights=0.3)
# S1STN_caud.connect_from_matrix(S1STN_weights)

STNGPi_caud = ann.Projection(pre=STN_caud, post=GPi_caud, target='exc', name='STNGPi_caud')
STNGPi_weights = STN_GPi_connection(preDim=model_params['num_init_positions'],
                                    postDim=model_params['dim_medial_BG'],
                                    weight=1.0)
STNGPi_caud.connect_from_matrix(STNGPi_weights)

# connection to output nuclei
GPiVA = ann.Projection(pre=GPi_caud, post=VA, target='inh', name='GPiVA_caud')
GPiVA.connect_one_to_one(weights=0.8)

# DA release
PPTNSNc_caud = ann.Projection(pre=PPTN, post=SNc_caud, target='exc', name='PPTNSNc_caud')
PPTNSNc_caud.connect_one_to_one(weights=1.0)

# direct
SNcStrD1_caud = ann.Projection(pre=SNc_caud, post=StrD1_caud, target='dopa', name='SNcStrD1_caud')
SNcStrD1_caud.connect_all_to_all(weights=1.0)

SNcGPi_caud = ann.Projection(pre=SNc_caud, post=GPi_caud, target='dopa', name='SNcGPi_caud')
SNcGPi_caud.connect_all_to_all(weights=1.0)

# indirect
SNcStrD2_caud = ann.Projection(pre=SNc_caud, post=StrD2_caud, target='dopa', name='SNcStrD2_caud')
SNcStrD2_caud.connect_all_to_all(weights=1.0)

SNcGPe_caud = ann.Projection(pre=SNc_caud, post=GPe_caud, target='dopa', name='SNcGPe_caud')
SNcGPe_caud.connect_all_to_all(weights=1.0)

# hyperdirect
SNcSTN_caud = ann.Projection(pre=SNc_caud, post=STN_caud, target='dopa', name='SNcSTN_caud')
SNcSTN_caud.connect_all_to_all(weights=1.0)

# DA prediction (tau can be higher)
StrD1SNc_caud = ann.Projection(pre=StrD1_caud, post=SNc_caud, target='inh', synapse=DAPrediction, name='StrD1SNc_caud')
StrD1SNc_caud.connect_all_to_all(weights=0.0)
StrD1SNc_caud.tau = 3000

# FB connections (all weights not fitted)
VAStrThal = ann.Projection(pre=VA, post=StrThal_caud, target='exc', name='VAStrThal')
VAStrThal.connect_one_to_one(weights=1.0)

StrThalGPi_caud = ann.Projection(pre=StrThal_caud, post=GPi_caud, target='inh', name='StrThalGPi_caud')
StrThalGPi_caud.connect_one_to_one(weights=0.1)

StrThalGPe_caud = ann.Projection(pre=StrThal_caud, post=GPe_caud, target='inh', name='StrThalGPe_caud')
StrThalGPe_caud.connect_one_to_one(weights=0.5)

# Lateral connections
# Striatum
StrD2StrD2_caud = ann.Projection(pre=StrD2_caud, post=StrD2_caud, target='inh')
StrD2StrD2_caud.connect_all_to_all(weights=0.1)

StrD1StrD1_caud = ann.Projection(pre=StrD1_caud, post=StrD1_caud, target='inh')
StrD1StrD1_caud.connect_all_to_all(weights=0.4)

# other laterals layerwise
w_caud_laterals = laterals_layerwise(preDim=model_params['dim_medial_BG'], postDim=model_params['dim_medial_BG'])

StrThalStrThal_caud = ann.Projection(pre=StrThal_caud, post=StrThal_caud, target='inh')
StrThalStrThal_caud.connect_from_matrix(0.3 * w_caud_laterals) #0.5

GPiGPi = ann.Projection(pre=GPi_caud, post=GPi_caud, target='exc', synapse=ReversedSynapse)
GPiGPi.connect_from_matrix(0.15 * w_caud_laterals)
GPiGPi.reversal = 0.4

GPeGPe = ann.Projection(pre=GPe_caud, post=GPe_caud, target='inh')
GPeGPe.connect_from_matrix(0.1 * w_caud_laterals)

VAVA = ann.Projection(pre=VA, post=VA, target='inh')
VAVA.connect_from_matrix(0.1 * w_caud_laterals)

