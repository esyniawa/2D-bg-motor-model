import ANNarchy as ann
import numpy as np

from model_definitions import *
from parameters import model_params

''' Basal Ganglia network to integrate a goal depending on the body schema'''

dPFC = ann.Population(name="dPFC", geometry=model_params['num_goals'], neuron=BaselineNeuron)

StrD1_caud = ann.Population(name="StrD1", geometry=model_params['dim_medial_Str'], neuron=LinearNeuron_trace)
StrD1_caud.noise = 0.01
StrD1_caud.lesion = 1.0

StrD2_caud = ann.Population(name="StrD2_caud", geometry=model_params['dim_medial_Str'], neuron=LinearNeuron_trace)
StrD2_caud.noise = 0.01
StrD2_caud.lesion = 1.0

STN_caud = ann.Population(name="STN_caud", geometry=model_params['num_init_positions'], neuron=BaselineNeuron)
STN_caud.noise = 0.01

GPi = ann.Population(name="GPi", geometry=model_params['dim_medial_BG'], neuron=LinearNeuron)
GPi.noise = 0.05
GPi.baseline = 1.9

GPe = ann.Population(name="GPe", geometry=model_params['dim_medial_BG'], neuron=LinearNeuron)
GPe.noise = 0.05
GPe.baseline = 1.0

VA = ann.Population(name="Thal_caud", geometry=model_params['dim_medial_BG'], neuron=LinearNeuron)
VA.noise = 0.025
VA.baseline = 0.7

StrThal_caud = ann.Population(name="StrThal_caud", geometry=model_params['dim_medial_BG'], neuron=LinearNeuron)
StrThal_caud.noise = 0.01
StrThal_caud.baseline = 0.4

SNc_caud = ann.Population(name='SNc_caud', geometry=model_params['num_goals'], neuron=DopamineNeuron)
SNc_caud.exc_threshold = 0.2
SNc_caud.baseline = 0.1
SNc_caud.factor_inh = 1.0

# Projections
PFCdStrD1 = ann.Projection(pre=dPFC, post=StrD1_caud, target='exc', synapse=DAPostCovarianceNoThreshold)
PFCdStrD1.connect_all_to_all(weights=ann.Normal(0.5, 0.2))
PFCdStrD1.tau = 400 #100
PFCdStrD1.regularization_threshold = 2.0
PFCdStrD1.tau_alpha = 5.0
PFCdStrD1.baseline_dopa = 8*baseline_dopa_caud
PFCdStrD1.K_dip = 0.05
PFCdStrD1.K_burst = 1.0
PFCdStrD1.DA_type = 1
PFCdStrD1.threshold_pre = 0.2
PFCdStrD1.threshold_post = 0.0

PFCdStrD2_caud = ann.Projection(pre=Objectives,post=StrD2_caud,target='exc', synapse=DAPostCovarianceNoThreshold)
PFCdStrD2_caud.connect_all_to_all(weights = ann.Normal(0.12, 0.03)) #0.005
PFCdStrD2_caud.tau = 2000.0
PFCdStrD2_caud.regularization_threshold = 1.5
PFCdStrD2_caud.tau_alpha = 15.0
PFCdStrD2_caud.baseline_dopa = 0.1
PFCdStrD2_caud.K_dip = 0.2
PFCdStrD2_caud.K_burst = 1.0
PFCdStrD2_caud.DA_type = -1
PFCdStrD2_caud.threshold_pre = 0.05
PFCdStrD2_caud.threshold_post = 0.05

StrD1GPi = ann.Projection(pre=StrD1_caud, post=GPi, target='inh', synapse=DAPreCovariance_inhibitory_trace)
StrD1GPi.connect_all_to_all(weights=ann.Normal(0.1, 0.01)) # scale by numer of stimuli #var 0.01
StrD1GPi.tau = 1600 #700 #550
StrD1GPi.regularization_threshold = 2.25 #1.5
StrD1GPi.tau_alpha = 4.0 # 20.0
StrD1GPi.baseline_dopa = 0.1
StrD1GPi.K_dip = 0.9
StrD1GPi.K_burst = 1.0
StrD1GPi.threshold_post = 0.3 #0.1 #0.3
StrD1GPi.threshold_pre = 0.05 # 0.15
StrD1GPi.DA_type = 1

StrD2GPe = ann.Projection(pre=StrD2_caud, post=GPe, target='inh', synapse=DAPreCovariance_inhibitory_trace)
StrD2GPe.connect_all_to_all(weights=0.01)
StrD2GPe.tau = 2500
StrD2GPe.regularization_threshold = 1.5
StrD2GPe.tau_alpha = 20.0
StrD2GPe.baseline_dopa = 0.1
StrD2GPe.K_dip = 0.1
StrD2GPe.K_burst = 1.2
StrD2GPe.threshold_post = 0.0
StrD2GPe.threshold_pre = 0.1
StrD2GPe.DA_type = -1

