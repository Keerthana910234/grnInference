import numpy as np
from numpy import random
import pandas as pd
import os.path
from numba import jit, njit
from numba import NumbaWarning
import warnings
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
warnings.simplefilter('ignore', category=NumbaWarning)

random.seed()

state_vars_both_TF = [
    "Target_is_bursting",
    "TF_is_bursting",
    "TF_protein_1K",
    "Target_protein_1K",
    "spliced_labeled_Target",
    "spliced_labeled_TF",
    "spliced_unlabeled_Target",
    "spliced_unlabeled_TF",
    "unspliced_labeled_Target",
    "unspliced_labeled_TF",
    "unspliced_unlabeled_Target",
    "unspliced_unlabeled_TF",
    "unspliced_Target",
    "mRNA_ever_produced_Target",
    "mRNA_ever_produced_TF",
    "protein_ever_produced_TF",
    "k_on_Target_adjusted",
    "total_TF_mRNA",
    "total_Target_mRNA",
    "protein_ever_produced_Target" ,
    "k_on_TF_adjusted"
]

state_vars = [
    "Target_is_bursting",
    "TF_is_bursting",
    "TF_protein_1K",
    "spliced_labeled_Target",
    "spliced_labeled_TF",
    "spliced_unlabeled_Target",
    "spliced_unlabeled_TF",
    "unspliced_labeled_Target",
    "unspliced_labeled_TF",
    "unspliced_unlabeled_Target",
    "unspliced_unlabeled_TF",
    "unspliced_Target",
    "mRNA_ever_produced_Target",
    "mRNA_ever_produced_TF",
    "protein_ever_produced_TF",
    "k_on_Target_adjusted",
    "total_TF_mRNA",
    "total_Target_mRNA"
]

#####################################################################################################################################
#Modifying simulation where both are TF
#####################################################################################################################################

def get_new_state_both_TF(in_pulse=None,
                  Target_is_bursting=None,
                  TF_is_bursting=None,
                  TF_protein_1K=None,
                  Target_protein_1K = None,
                  spliced_labeled_Target=None,
                  spliced_labeled_TF=None,
                  spliced_unlabeled_Target=None,
                  spliced_unlabeled_TF=None,
                  unspliced_labeled_Target=None,
                  unspliced_labeled_TF=None,
                  unspliced_unlabeled_Target=None,
                  unspliced_unlabeled_TF=None,
                  unspliced_Target=None,
                  TF_transcription_rate=None,
                  Target_transcription_rate=None,
                  TF_mRNA_degradation_rate=None,
                  Target_mRNA_degradation_rate=None,
                  labeling_efficiency=None,
                  capture_efficiency=None,
                  splicing_rate=None,
                  TF_protein_production_rate=None,
                  Target_protein_production_rate=None,
                  TF_protein_degradation_rate=None,
                  Target_protein_degradation_rate=None,
                  k_on_TF=None,
                  k_off_TF=None,
                  k_on_Target=None,
                  k_off_Target=None,
                  TF_Target_link_EC50=None,
                  Target_TF_link_EC50=None,
                  total_TF_mRNA=None,
                  total_Target_mRNA=None,
                  k_on_TF_adjusted=None,
                  k_on_Target_adjusted=None,
                  mRNA_ever_produced_Target=None,
                  mRNA_ever_produced_TF=None,
                  protein_ever_produced_TF=None,
                  protein_ever_produced_Target=None,
                  n=None,
                  t=1 / 60):
    #t = 1/60 since all rates are in hours
    #####################################
    
#   degrade before the continuous calculations
#   degradation of TF mRNA
    unspliced_unlabeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, unspliced_unlabeled_TF, t)
    unspliced_unlabeled_TF = np.maximum(unspliced_unlabeled_TF, 0)
    spliced_unlabeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, spliced_unlabeled_TF, t)
    spliced_unlabeled_TF = np.maximum(spliced_unlabeled_TF, 0)
    
    unspliced_labeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, unspliced_labeled_TF, t)
    unspliced_labeled_TF = np.maximum(unspliced_labeled_TF, 0)
    spliced_labeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, spliced_labeled_TF, t)
    spliced_labeled_TF = np.maximum(spliced_labeled_TF, 0)
   
    # degradation of TF protein
    TF_protein_1K -= degrade_poisson(TF_protein_degradation_rate, TF_protein_1K * 1000, t) / 1000
    TF_protein_1K = np.maximum(TF_protein_1K, 0)

    # degradation of Target mRNA
    unspliced_unlabeled_Target -= degrade_poisson(Target_mRNA_degradation_rate, unspliced_unlabeled_Target, t)
    unspliced_unlabeled_Target = np.maximum(unspliced_unlabeled_Target, 0)
    spliced_unlabeled_Target -= degrade_poisson(Target_mRNA_degradation_rate, spliced_unlabeled_Target, t)
    spliced_unlabeled_Target = np.maximum(spliced_unlabeled_Target, 0)
    
    unspliced_labeled_Target -= degrade_poisson(Target_mRNA_degradation_rate, unspliced_labeled_Target, t)
    unspliced_labeled_Target = np.maximum(unspliced_labeled_Target, 0)
    spliced_labeled_Target -= degrade_poisson(Target_mRNA_degradation_rate, spliced_labeled_Target, t)
    spliced_labeled_Target = np.maximum(spliced_labeled_Target, 0)

    # degradation of Target protein
    Target_protein_1K -= degrade_poisson(Target_protein_degradation_rate, Target_protein_1K * 1000, t) / 1000
    Target_protein_1K = np.maximum(Target_protein_1K, 0)
    
    #####################################

    num_cells = TF_protein_1K.shape[0]
    
    ####TF changes####
    
    # absolute new TF mRNA is a function of txn rate and whether TF is currently bursting
    new_TF_mRNA = TF_transcription_rate * TF_is_bursting 
    D_mRNA_ever_produced_TF = new_TF_mRNA.copy()

    # labeling
    # labeling efficiency only matters if in pulse (binary event)
    D_unspliced_labeled_TF = new_TF_mRNA * labeling_efficiency * in_pulse
    D_unspliced_unlabeled_TF = new_TF_mRNA * (1 - labeling_efficiency * in_pulse)

    #note: splicing rate (immature->mature mRNA) is in minutes
    # splicing
    # labeled
    D_unspliced_labeled_TF -= splicing_rate * unspliced_labeled_TF
    D_spliced_labeled_TF = splicing_rate * unspliced_labeled_TF
    # unlabeled
    D_unspliced_unlabeled_TF -= splicing_rate * unspliced_unlabeled_TF
    D_spliced_unlabeled_TF = splicing_rate * unspliced_unlabeled_TF

    # protein
    all_spliced_TF = spliced_unlabeled_TF + spliced_labeled_TF
    D_TF_protein_1K = all_spliced_TF * TF_protein_production_rate
    D_protein_ever_produced_TF = D_TF_protein_1K.copy()

    ####TF state update####
    #for timestep t (default = 1min)
    TF_protein_1K = TF_protein_1K + D_TF_protein_1K * t
    spliced_labeled_TF = spliced_labeled_TF + D_spliced_labeled_TF * t
    spliced_unlabeled_TF = spliced_unlabeled_TF + D_spliced_unlabeled_TF * t
    unspliced_labeled_TF = unspliced_labeled_TF + D_unspliced_labeled_TF * t
    unspliced_unlabeled_TF = unspliced_unlabeled_TF + D_unspliced_unlabeled_TF * t
    mRNA_ever_produced_TF = mRNA_ever_produced_TF + D_mRNA_ever_produced_TF * t
    protein_ever_produced_TF = protein_ever_produced_TF + D_protein_ever_produced_TF*t
    
    ####Target changes####
    
    # absolute new Target mRNA is a function of txn rate and whether Target is currently bursting
    new_Target_mRNA = Target_transcription_rate * Target_is_bursting
    D_mRNA_ever_produced_Target = new_Target_mRNA.copy()

    # labeling
    # labeling efficiency only matters if in pulse (binary event)
    D_unspliced_labeled_Target = new_Target_mRNA * labeling_efficiency * in_pulse
    D_unspliced_unlabeled_Target = new_Target_mRNA * (1 - labeling_efficiency * in_pulse)

    # splicing
    # labeled
    D_unspliced_labeled_Target -= splicing_rate * unspliced_labeled_Target
    D_spliced_labeled_Target = splicing_rate * unspliced_labeled_Target
    # unlabeled
    D_unspliced_unlabeled_Target -= splicing_rate * unspliced_unlabeled_Target
    D_spliced_unlabeled_Target = splicing_rate * unspliced_unlabeled_Target
    # protein
    all_spliced_Target = spliced_unlabeled_Target + spliced_labeled_Target
    D_Target_protein_1K = all_spliced_Target * Target_protein_production_rate
    D_Target_protein_1K = np.maximum(D_Target_protein_1K, 0)
    D_protein_ever_produced_Target = D_Target_protein_1K.copy()

    ####Target state update####
    #w timestep t (default=1 min)
    spliced_labeled_Target = spliced_labeled_Target + D_spliced_labeled_Target * t
    spliced_unlabeled_Target = spliced_unlabeled_Target + D_spliced_unlabeled_Target * t
    unspliced_labeled_Target = unspliced_labeled_Target + D_unspliced_labeled_Target * t
    unspliced_unlabeled_Target = unspliced_unlabeled_Target + D_unspliced_unlabeled_Target * t
    mRNA_ever_produced_Target = mRNA_ever_produced_Target + D_mRNA_ever_produced_Target * t
    Target_protein_1K = Target_protein_1K + D_Target_protein_1K * t
    protein_ever_produced_Target = protein_ever_produced_Target + D_Target_protein_1K * t

    ################################################################################
    ## ----> CHECK THAT LINK FUNCTION IS ON IF NOT DOING STATE-BASED CHECKS <---- ##
    ################################################################################
    #Regulation effects
    # k_on_Target_adjusted = k_on_Target * 1
    k_on_Target_adjusted = k_on_Target * (1 + TF_Target_link_function(TF_protein_1K, TF_Target_link_EC50, n, 16))
    k_on_TF_adjusted = k_on_TF*(1 + TF_Target_link_function(Target_protein_1K, Target_TF_link_EC50, n, 16))
    
    Target_should_switch_off = Target_is_bursting & (np.random.exponential(1 / k_off_Target, num_cells) < t)
    Target_should_switch_on = (~ Target_is_bursting) & (np.random.exponential(1 / k_on_Target_adjusted, num_cells) < t)

    TF_should_switch_off = TF_is_bursting & (np.random.exponential(1 / k_off_TF, num_cells) < t)
    TF_should_switch_on = (~ TF_is_bursting) & (np.random.exponential(1 / k_on_TF_adjusted, num_cells) < t)

    # update whether TF is bursting in this new state
    new_TF_bursting = TF_is_bursting | TF_should_switch_on #It remains on or turned on
    new_TF_bursting[TF_should_switch_off] = False #Turn off whatever was on but should turn off
    TF_is_bursting = new_TF_bursting

    new_Target_bursting = Target_is_bursting | Target_should_switch_on
    new_Target_bursting[Target_should_switch_off] = False
    Target_is_bursting = new_Target_bursting

    total_TF_mRNA = spliced_unlabeled_TF + spliced_labeled_TF + unspliced_unlabeled_TF + unspliced_labeled_TF
    total_Target_mRNA = unspliced_unlabeled_Target + spliced_unlabeled_Target + unspliced_labeled_Target + spliced_labeled_Target
    
    unspliced_TF = unspliced_labeled_TF + unspliced_unlabeled_TF
    unspliced_Target = unspliced_labeled_Target + unspliced_unlabeled_Target

    return {
        "TF_is_bursting": TF_is_bursting,
        "Target_is_bursting": Target_is_bursting,
        "TF_protein_1K": TF_protein_1K,
        "Target_protein_1K": Target_protein_1K,
        "spliced_labeled_Target": spliced_labeled_Target,
        "spliced_labeled_TF": spliced_labeled_TF,
        "spliced_unlabeled_Target": spliced_unlabeled_Target,
        "spliced_unlabeled_TF": spliced_unlabeled_TF,
        "unspliced_labeled_Target": unspliced_labeled_Target,
        "unspliced_labeled_TF": unspliced_labeled_TF,
        "unspliced_unlabeled_Target": unspliced_unlabeled_Target,
        "unspliced_unlabeled_TF": unspliced_unlabeled_TF,
        "unspliced_Target": unspliced_Target,
        "mRNA_ever_produced_Target": mRNA_ever_produced_Target,
        "mRNA_ever_produced_TF": mRNA_ever_produced_TF,
        "protein_ever_produced_TF": protein_ever_produced_TF,
        "protein_ever_produced_Target": protein_ever_produced_Target,
        "k_on_Target_adjusted": k_on_Target_adjusted,
        "k_on_TF_adjusted": k_on_TF_adjusted,
        "total_TF_mRNA": total_TF_mRNA,
        "total_Target_mRNA": total_Target_mRNA}

@njit(fastmath=True)
def TF_Target_link_function(tf_val, EC_50, Hill_coef, max_effect):
        return max_effect * (tf_val**Hill_coef) / ((EC_50**Hill_coef) + (tf_val**Hill_coef))

@njit
def MM_TF_link(TF_mean, Hill_coef, max_effect):
    EC_50 = np.power(max_effect * TF_mean**Hill_coef - TF_mean**Hill_coef, 1.0/Hill_coef)
    # return EC_50
    return TF_mean

#to get TF protein mean for TF protein->Target k_on link function
#to get TF protein mean for TF protein->Target k_on link function
def get_steady_state_self_regulating_TF(
    k_on_TF, k_off_TF, TF_transcription_rate, TF_mRNA_degradation_rate, 
    splicing_rate, TF_protein_production_rate, TF_protein_degradation_rate,
    k_on_Target, k_off_Target, Target_transcription_rate, Target_mRNA_degradation_rate, 
    Target_protein_production_rate, Target_protein_degradation_rate,
    max_effect=16.0, TF_hill=0.5, **kwargs
):
    """
    Calculate steady state for system where:
    - TF regulates itself (self-regulation)
    - TF regulates Target 
    - Hill function = 0.5 at steady state (EC50 normalization)
    
    Returns steady-state protein levels for TF and Target
    """
    
    def steady_state_equations(TF_p_ss):
        """
        System of equations to solve for TF steady state.
        At steady state, Hill function = 0.5 by design.
        """
        
        # TF self-regulation: Hill function = 0.5 at steady state
        k_TF_eff = k_on_TF * max_effect * TF_hill  # TF_hill = 0.5
        b_TF = k_TF_eff / (k_TF_eff + k_off_TF)
        
        # Calculate what TF protein level should be given this burst fraction
        TF_u_ss = (TF_transcription_rate * b_TF) / (TF_mRNA_degradation_rate + splicing_rate)
        TF_s_ss = (TF_u_ss * splicing_rate) / TF_mRNA_degradation_rate
        TF_p_predicted = (TF_s_ss * TF_protein_production_rate) / TF_protein_degradation_rate
        
        # This should equal our input TF_p_ss for self-consistency
        return TF_p_predicted - TF_p_ss
    
    # Solve for self-consistent TF protein level
    TF_p_ss = fsolve(steady_state_equations, 1.0)[0]
    
    # Now calculate Target steady state (regulated by TF)
    # Target is regulated by TF protein level
    k_Target_eff = k_on_Target * max_effect * TF_hill  # Same Hill value (0.5)
    b_Target = k_Target_eff / (k_Target_eff + k_off_Target)
    
    # Target steady states
    Target_u_ss = (Target_transcription_rate * b_Target) / (Target_mRNA_degradation_rate + splicing_rate)
    Target_s_ss = (Target_u_ss * splicing_rate) / Target_mRNA_degradation_rate
    Target_p_ss = (Target_s_ss * Target_protein_production_rate) / Target_protein_degradation_rate
    
    # Recalculate TF intermediates for completeness
    k_TF_eff = k_on_TF * max_effect * TF_hill
    b_TF = k_TF_eff / (k_TF_eff + k_off_TF)
    TF_u_ss = (TF_transcription_rate * b_TF) / (TF_mRNA_degradation_rate + splicing_rate)
    TF_s_ss = (TF_u_ss * splicing_rate) / TF_mRNA_degradation_rate
    
    return {
        'average_protein_TF': TF_p_ss,
        'average_protein_Target': Target_p_ss,
    }

    average_val = {}
    average_val['average_protein_TF'] = sol.y[2, -1]  # TF protein
    average_val['average_protein_Target'] = sol.y[5, -1]  # Target protein
    return average_val

def generate_random_cells_both_TF(mean_TF_protein, mean_Target_protein, num_cells):
    state = {k: np.zeros(int(num_cells)).astype(np.bool if 'is_bursting' in k else np.float64) for k in state_vars_both_TF}
    # seed TF concentration with a random value - use to check for mixing
    state['TF_protein_1K'] = np.random.uniform(0, 3 * mean_TF_protein, int(num_cells))
    state['Target_protein_1K'] = np.random.uniform(0, 3 * mean_Target_protein, int(num_cells))
    return state

#to ensure cells are in steady-state
def burn_in_TF_concentration_both_TF(constants, num_cells=500, mean_TF_protein=20, mean_Target_protein=20, days=12):
    state = generate_random_cells_both_TF(mean_TF_protein, mean_Target_protein, num_cells=num_cells)
    for day in range(days):
        for steps in range(60 * 24): #This means that burn in time is 6 days - adjust accordingly
            state = get_new_state_both_TF(**state, **constants)
    return state['TF_protein_1K'].mean(), state['TF_protein_1K'].std(), state['Target_protein_1K'].mean(), state['Target_protein_1K'].std()

#Simulation function for synchronous division
def simulate_both_TF(**sim_args):
    random.seed()
    constants = sim_args.copy()
    # all rates are in hours
    constants.update({
        'TF_mRNA_degradation_rate': np.log(2) / sim_args['mrna_half_life_TF'],
        'Target_mRNA_degradation_rate': np.log(2) / sim_args['mrna_half_life_Target'],
        'TF_transcription_rate': sim_args['burst_size_TF'] * sim_args['k_off_TF'],
        'Target_transcription_rate': sim_args['burst_size_Target'] * sim_args['k_off_Target'],
        'TF_protein_degradation_rate': np.log(2) / sim_args['TF_protein_half_life'],
        'Target_protein_degradation_rate': np.log(2) / sim_args['Target_protein_half_life'],
        'splicing_rate': np.log(2) / (sim_args['splicing_half_life_minutes'] / 60),
        'in_pulse': 0, 'n': sim_args['n'],
        't': 1/60,
    })
    for k, v in constants.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.integer):
            constants[k] = v.astype(float)

    constants['TF_Target_link_EC50'] = 1
    dynamics = 'Michaelis-Menton'
    pulse_time = sim_args['pulse_time']
    num_cells = sim_args['num_cells']

    # trim unused settings
    for unused in {'mrna_half_life_TF', 'mrna_half_life_Target', 'TF_protein_half_life',
                    'Target_protein_half_life', 'splicing_half_life_minutes', 'dynamics', 
                    'pulse_time', 'num_cells', 'burst_size_TF', 'burst_size_Target'}:
        del constants[unused]

    # Find TF mean through calculation
    averages = get_average_numbers_both_TF(**constants)
    average_protein_TF = averages['average_protein_TF']
    average_protein_Target = averages['average_protein_Target']

    constants['TF_Target_link_EC50'] = MM_TF_link(average_protein_TF, sim_args['n'], 16)
    constants['Target_TF_link_EC50'] = MM_TF_link(average_protein_Target, sim_args['n'], 16)
    
    # burn in TF concentrations for variance
    print("burning in")
    TF_protein_mean, TF_protein_std, Target_protein_mean, Target_protein_std = burn_in_TF_concentration_both_TF(constants, mean_TF_protein=average_protein_TF, mean_Target_protein=average_protein_Target)
    print("simulating")

    state = generate_random_cells_both_TF(TF_protein_mean, Target_protein_mean, num_cells=num_cells)
    initial_TF = state['TF_protein_1K']
    initial_Target = state['Target_protein_1K']

    samples = {}
    samplesReplicate = {}
    days = 13 #to allow for 1 day after pulse start during which we can sample
    
    # when does pulse happen and for how long?
    day_to_start_pulse_at = 12
    hours_to_track_for = 24
    pulse_start = day_to_start_pulse_at * 60 * hours_to_track_for
    # pulse_start = 0
    pulse_end = int(pulse_start + pulse_time)
    twin_divide_time = 12*60*24 #in minutes - start after reaching steady state
    assert pulse_end < days * 24 * 60
    
    for steps in range(60 * 24 * days + 1):
        if (pulse_start < steps < pulse_end):
            constants['in_pulse'] = 1
        else:
            constants['in_pulse'] = 0
        if (steps == twin_divide_time):
            replicateState = state.copy()
        state = get_new_state_both_TF(**state, **constants)

        for k in state:
            if k != 'TF_Target_link_EC50' and k != 'Target_TF_link_EC50':
                assert np.all(state[k] >= -0), k
        if steps >= twin_divide_time:
            replicateState = get_new_state_both_TF(**replicateState, **constants)
            for k in replicateState:
                if k != 'TF_Target_link_EC50' and k != 'Target_TF_link_EC50':
                    assert np.all(state[k] >= -0), k
                
        time_since_start = steps - pulse_start #to sample from start of pulse instead of end
        sampling_interval = 60 #in minutes
        if ((time_since_start >= 0) and (time_since_start % sampling_interval == 0)):
            samples[time_since_start] = state
            samplesReplicate[time_since_start] = replicateState
          
    # convert samples to dataframes
    for k in samples:
        samples[k] = pd.DataFrame(samples[k])
        samples[k]['sampling_time'] = k
    
    for k in samplesReplicate:
        samplesReplicate[k] = pd.DataFrame(samplesReplicate[k])
        samplesReplicate[k]['sampling_time'] = k

    # prepare TF calibration averages for reporting
    TF_values = {k.replace('average_', 'mean TF '): "{:.2f}".format(v) for k, v in averages.items()}

    # prepare constants for reporting
    del constants['in_pulse']
    del constants['TF_Target_link_EC50']
    constants['dynamics'] = dynamics
    constants['pulse time (minutes)'] = pulse_time

    all_samples = pd.concat(samples, ignore_index=True)
    all_samples['replicate'] = 0
    all_samplesReplicate = pd.concat(samplesReplicate, ignore_index=True)
    all_samplesReplicate['replicate'] = 1
    all_samples_both_replicates = pd.concat([all_samples, all_samplesReplicate], ignore_index=True)
    return all_samples_both_replicates

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#Simulate self-regulation
def get_new_state_self_regulation(in_pulse=None,
                  Target_is_bursting=None,
                  TF_is_bursting=None,
                  TF_protein_1K=None,
                  Target_protein_1K = None,
                  spliced_labeled_Target=None,
                  spliced_labeled_TF=None,
                  spliced_unlabeled_Target=None,
                  spliced_unlabeled_TF=None,
                  unspliced_labeled_Target=None,
                  unspliced_labeled_TF=None,
                  unspliced_unlabeled_Target=None,
                  unspliced_unlabeled_TF=None,
                  unspliced_Target=None,
                  TF_transcription_rate=None,
                  Target_transcription_rate=None,
                  TF_mRNA_degradation_rate=None,
                  Target_mRNA_degradation_rate=None,
                  labeling_efficiency=None,
                  capture_efficiency=None,
                  splicing_rate=None,
                  TF_protein_production_rate=None,
                  Target_protein_production_rate=None,
                  TF_protein_degradation_rate=None,
                  Target_protein_degradation_rate=None,
                  k_on_TF=None,
                  k_off_TF=None,
                  k_on_Target=None,
                  k_off_Target=None,
                  TF_Target_link_EC50=None,
                  Target_TF_link_EC50=None,
                  total_TF_mRNA=None,
                  total_Target_mRNA=None,
                  k_on_TF_adjusted=None,
                  k_on_Target_adjusted=None,
                  mRNA_ever_produced_Target=None,
                  mRNA_ever_produced_TF=None,
                  protein_ever_produced_TF=None,
                  protein_ever_produced_Target=None,
                  n=None,
                  t=1 / 60):
    #t = 1/60 since all rates are in hours
    #####################################
    
#   degrade before the continuous calculations
#   degradation of TF mRNA
    unspliced_unlabeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, unspliced_unlabeled_TF, t)
    unspliced_unlabeled_TF = np.maximum(unspliced_unlabeled_TF, 0)
    spliced_unlabeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, spliced_unlabeled_TF, t)
    spliced_unlabeled_TF = np.maximum(spliced_unlabeled_TF, 0)
    
    unspliced_labeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, unspliced_labeled_TF, t)
    unspliced_labeled_TF = np.maximum(unspliced_labeled_TF, 0)
    spliced_labeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, spliced_labeled_TF, t)
    spliced_labeled_TF = np.maximum(spliced_labeled_TF, 0)
   
    # degradation of TF protein
    TF_protein_1K -= degrade_poisson(TF_protein_degradation_rate, TF_protein_1K * 1000, t) / 1000
    TF_protein_1K = np.maximum(TF_protein_1K, 0)

    # degradation of Target mRNA
    unspliced_unlabeled_Target -= degrade_poisson(Target_mRNA_degradation_rate, unspliced_unlabeled_Target, t)
    unspliced_unlabeled_Target = np.maximum(unspliced_unlabeled_Target, 0)
    spliced_unlabeled_Target -= degrade_poisson(Target_mRNA_degradation_rate, spliced_unlabeled_Target, t)
    spliced_unlabeled_Target = np.maximum(spliced_unlabeled_Target, 0)
    
    unspliced_labeled_Target -= degrade_poisson(Target_mRNA_degradation_rate, unspliced_labeled_Target, t)
    unspliced_labeled_Target = np.maximum(unspliced_labeled_Target, 0)
    spliced_labeled_Target -= degrade_poisson(Target_mRNA_degradation_rate, spliced_labeled_Target, t)
    spliced_labeled_Target = np.maximum(spliced_labeled_Target, 0)

    # degradation of Target protein
    Target_protein_1K -= degrade_poisson(Target_protein_degradation_rate, Target_protein_1K * 1000, t) / 1000
    Target_protein_1K = np.maximum(Target_protein_1K, 0)
    
    #####################################

    num_cells = TF_protein_1K.shape[0]
    
    ####TF changes####
    
    # absolute new TF mRNA is a function of txn rate and whether TF is currently bursting
    new_TF_mRNA = TF_transcription_rate * TF_is_bursting 
    D_mRNA_ever_produced_TF = new_TF_mRNA.copy()

    # labeling
    # labeling efficiency only matters if in pulse (binary event)
    D_unspliced_labeled_TF = new_TF_mRNA * labeling_efficiency * in_pulse
    D_unspliced_unlabeled_TF = new_TF_mRNA * (1 - labeling_efficiency * in_pulse)

    #note: splicing rate (immature->mature mRNA) is in minutes
    # splicing
    # labeled
    D_unspliced_labeled_TF -= splicing_rate * unspliced_labeled_TF
    D_spliced_labeled_TF = splicing_rate * unspliced_labeled_TF
    # unlabeled
    D_unspliced_unlabeled_TF -= splicing_rate * unspliced_unlabeled_TF
    D_spliced_unlabeled_TF = splicing_rate * unspliced_unlabeled_TF

    # protein
    all_spliced_TF = spliced_unlabeled_TF + spliced_labeled_TF
    D_TF_protein_1K = all_spliced_TF * TF_protein_production_rate
    D_protein_ever_produced_TF = D_TF_protein_1K.copy()

    ####TF state update####
    #for timestep t (default = 1min)
    TF_protein_1K = TF_protein_1K + D_TF_protein_1K * t
    spliced_labeled_TF = spliced_labeled_TF + D_spliced_labeled_TF * t
    spliced_unlabeled_TF = spliced_unlabeled_TF + D_spliced_unlabeled_TF * t
    unspliced_labeled_TF = unspliced_labeled_TF + D_unspliced_labeled_TF * t
    unspliced_unlabeled_TF = unspliced_unlabeled_TF + D_unspliced_unlabeled_TF * t
    mRNA_ever_produced_TF = mRNA_ever_produced_TF + D_mRNA_ever_produced_TF * t
    protein_ever_produced_TF = protein_ever_produced_TF + D_protein_ever_produced_TF*t
    
    ####Target changes####
    
    # absolute new Target mRNA is a function of txn rate and whether Target is currently bursting
    new_Target_mRNA = Target_transcription_rate * Target_is_bursting
    D_mRNA_ever_produced_Target = new_Target_mRNA.copy()

    # labeling
    # labeling efficiency only matters if in pulse (binary event)
    D_unspliced_labeled_Target = new_Target_mRNA * labeling_efficiency * in_pulse
    D_unspliced_unlabeled_Target = new_Target_mRNA * (1 - labeling_efficiency * in_pulse)

    # splicing
    # labeled
    D_unspliced_labeled_Target -= splicing_rate * unspliced_labeled_Target
    D_spliced_labeled_Target = splicing_rate * unspliced_labeled_Target
    # unlabeled
    D_unspliced_unlabeled_Target -= splicing_rate * unspliced_unlabeled_Target
    D_spliced_unlabeled_Target = splicing_rate * unspliced_unlabeled_Target
    # protein
    all_spliced_Target = spliced_unlabeled_Target + spliced_labeled_Target
    D_Target_protein_1K = all_spliced_Target * Target_protein_production_rate
    D_Target_protein_1K = np.maximum(D_Target_protein_1K, 0)
    D_protein_ever_produced_Target = D_Target_protein_1K.copy()

    ####Target state update####
    #w timestep t (default=1 min)
    spliced_labeled_Target = spliced_labeled_Target + D_spliced_labeled_Target * t
    spliced_unlabeled_Target = spliced_unlabeled_Target + D_spliced_unlabeled_Target * t
    unspliced_labeled_Target = unspliced_labeled_Target + D_unspliced_labeled_Target * t
    unspliced_unlabeled_Target = unspliced_unlabeled_Target + D_unspliced_unlabeled_Target * t
    mRNA_ever_produced_Target = mRNA_ever_produced_Target + D_mRNA_ever_produced_Target * t
    Target_protein_1K = Target_protein_1K + D_Target_protein_1K * t
    protein_ever_produced_Target = protein_ever_produced_Target + D_Target_protein_1K * t

    ################################################################################
    ## ----> CHECK THAT LINK FUNCTION IS ON IF NOT DOING STATE-BASED CHECKS <---- ##
    ################################################################################
    #Regulation effects - self-regulation for TF and Target is regulated by TF
    # k_on_Target_adjusted = k_on_Target * 1
    k_on_Target_adjusted = k_on_Target * (1 + TF_Target_link_function(TF_protein_1K, TF_Target_link_EC50, n, 16) + TF_Target_link_function(Target_protein_1K, Target_TF_link_EC50, n, 16))
    k_on_TF_adjusted = k_on_TF* (1)
    # k_on_TF_adjusted = k_on_TF* (1 + TF_Target_link_function(TF_protein_1K, TF_Target_link_EC50, n, 16))
    
    Target_should_switch_off = Target_is_bursting & (np.random.exponential(1 / k_off_Target, num_cells) < t)
    Target_should_switch_on = (~ Target_is_bursting) & (np.random.exponential(1 / k_on_Target_adjusted, num_cells) < t)

    TF_should_switch_off = TF_is_bursting & (np.random.exponential(1 / k_off_TF, num_cells) < t)
    TF_should_switch_on = (~ TF_is_bursting) & (np.random.exponential(1 / k_on_TF_adjusted, num_cells) < t)

    # update whether TF is bursting in this new state
    new_TF_bursting = TF_is_bursting | TF_should_switch_on #It remains on or turned on
    new_TF_bursting[TF_should_switch_off] = False #Turn off whatever was on but should turn off
    TF_is_bursting = new_TF_bursting

    new_Target_bursting = Target_is_bursting | Target_should_switch_on
    new_Target_bursting[Target_should_switch_off] = False
    Target_is_bursting = new_Target_bursting

    total_TF_mRNA = spliced_unlabeled_TF + spliced_labeled_TF + unspliced_unlabeled_TF + unspliced_labeled_TF
    total_Target_mRNA = unspliced_unlabeled_Target + spliced_unlabeled_Target + unspliced_labeled_Target + spliced_labeled_Target
    
    unspliced_TF = unspliced_labeled_TF + unspliced_unlabeled_TF
    unspliced_Target = unspliced_labeled_Target + unspliced_unlabeled_Target

    return {
        "TF_is_bursting": TF_is_bursting,
        "Target_is_bursting": Target_is_bursting,
        "TF_protein_1K": TF_protein_1K,
        "Target_protein_1K": Target_protein_1K,
        "spliced_labeled_Target": spliced_labeled_Target,
        "spliced_labeled_TF": spliced_labeled_TF,
        "spliced_unlabeled_Target": spliced_unlabeled_Target,
        "spliced_unlabeled_TF": spliced_unlabeled_TF,
        "unspliced_labeled_Target": unspliced_labeled_Target,
        "unspliced_labeled_TF": unspliced_labeled_TF,
        "unspliced_unlabeled_Target": unspliced_unlabeled_Target,
        "unspliced_unlabeled_TF": unspliced_unlabeled_TF,
        "unspliced_Target": unspliced_Target,
        "mRNA_ever_produced_Target": mRNA_ever_produced_Target,
        "mRNA_ever_produced_TF": mRNA_ever_produced_TF,
        "protein_ever_produced_TF": protein_ever_produced_TF,
        "protein_ever_produced_Target": protein_ever_produced_Target,
        "k_on_Target_adjusted": k_on_Target_adjusted,
        "k_on_TF_adjusted": k_on_TF_adjusted,
        "total_TF_mRNA": total_TF_mRNA,
        "total_Target_mRNA": total_Target_mRNA}

def burn_in_TF_concentration_self_regulation(constants, num_cells=500, mean_TF_protein=20, mean_Target_protein=20, days=12):
    state = generate_random_cells_both_TF(mean_TF_protein, mean_Target_protein, num_cells=num_cells)
    for day in range(days):
        for steps in range(60 * 24): #This means that burn in time is 6 days - adjust accordingly
            state = get_new_state_self_regulation(**state, **constants)
    return state['TF_protein_1K'].mean(), state['TF_protein_1K'].std(), state['Target_protein_1K'].mean(), state['Target_protein_1K'].std()

# def get_average_numbers_self_regulation(
#     k_on_TF, k_off_TF, TF_transcription_rate, TF_mRNA_degradation_rate, 
#     splicing_rate, TF_protein_production_rate, TF_protein_degradation_rate,
#     k_on_Target, k_off_Target, Target_transcription_rate, Target_mRNA_degradation_rate, 
#     Target_protein_production_rate, Target_protein_degradation_rate,
#     max_effect=16.0, TF_hill=0.5, **kwargs
# ):
#     """
#     Calculate steady state for system where:
#     - TF regulates itself (self-regulation)
#     - TF regulates Target 
#     - Hill function = 0.5 at steady state (EC50 normalization)
    
#     Returns steady-state protein levels for TF and Target
#     """
    
#     def steady_state_equations(TF_p_ss):
#         """
#         System of equations to solve for TF steady state.
#         At steady state, Hill function = 0.5 by design.
#         """
        
#         # TF self-regulation: Hill function = 0.5 at steady state
#         k_TF_eff = k_on_TF *(1+ max_effect * TF_hill)  # TF_hill = 0.5
#         b_TF = k_TF_eff / (k_TF_eff + k_off_TF)
        
#         # Calculate what TF protein level should be given this burst fraction
#         TF_u_ss = (TF_transcription_rate * b_TF) / (TF_mRNA_degradation_rate + splicing_rate)
#         TF_s_ss = (TF_u_ss * splicing_rate) / TF_mRNA_degradation_rate
#         TF_p_predicted = (TF_s_ss * TF_protein_production_rate) / TF_protein_degradation_rate
        
#         # This should equal our input TF_p_ss for self-consistency
#         return TF_p_predicted - TF_p_ss
    
#     # Solve for self-consistent TF protein level
#     TF_p_ss = fsolve(steady_state_equations, 1.0)[0]
    
#     # Now calculate Target steady state (regulated by TF)
#     # Target is regulated by TF protein level
#     k_Target_eff = k_on_Target *(1 + max_effect * TF_hill)  # Same Hill value (0.5)
#     b_Target = k_Target_eff / (k_Target_eff + k_off_Target)
    
#     # Target steady states
#     Target_u_ss = (Target_transcription_rate * b_Target) / (Target_mRNA_degradation_rate + splicing_rate)
#     Target_s_ss = (Target_u_ss * splicing_rate) / Target_mRNA_degradation_rate
#     Target_p_ss = (Target_s_ss * Target_protein_production_rate) / Target_protein_degradation_rate
    
#     # Recalculate TF intermediates for completeness
#     k_TF_eff = k_on_TF *(1+ max_effect * TF_hill)
#     b_TF = k_TF_eff / (k_TF_eff + k_off_TF)
#     TF_u_ss = (TF_transcription_rate * b_TF) / (TF_mRNA_degradation_rate + splicing_rate)
#     TF_s_ss = (TF_u_ss * splicing_rate) / TF_mRNA_degradation_rate
    
#     return {
#         'average_protein_TF': TF_p_ss,
#         'average_protein_Target': Target_p_ss,
#     }

def get_average_numbers_self_loop_regulation(
    k_on_TF, k_off_TF, TF_transcription_rate, TF_mRNA_degradation_rate, 
    splicing_rate, TF_protein_production_rate, TF_protein_degradation_rate,
    k_on_Target, k_off_Target, Target_transcription_rate, Target_mRNA_degradation_rate, 
    Target_protein_production_rate, Target_protein_degradation_rate,
    max_effect=16.0, TF_hill=0.5, Target_hill=0.5, **kwargs
):
    """
    Calculate steady state for system where:
    - TF has no regulation (constitutive)
    - TF regulates Target 
    - Target regulates itself (self-loop)
    - Hill functions = 0.5 at steady state (EC50 normalization)
    
    Returns steady-state protein levels for TF and Target
    """
    from scipy.optimize import fsolve
    
    def steady_state_equations(protein_levels):
        """
        System of equations to solve for both TF and Target steady states.
        protein_levels = [TF_p_ss, Target_p_ss]
        """
        TF_p_ss, Target_p_ss = protein_levels
        
        # === TF (no regulation - constitutive) ===
        b_TF = k_on_TF / (k_on_TF + k_off_TF)
        TF_u_ss = (TF_transcription_rate * b_TF) / (TF_mRNA_degradation_rate + splicing_rate)
        TF_s_ss = (TF_u_ss * splicing_rate) / TF_mRNA_degradation_rate
        TF_p_predicted = (TF_s_ss * TF_protein_production_rate) / TF_protein_degradation_rate
        
        # === Target (regulated by TF + self-regulation) ===
        # Regulatory effects: TF→Target + Target→Target (self-loop)
        TF_effect = max_effect * TF_hill      # TF regulation effect (fixed at 0.5)
        Target_effect = max_effect * Target_hill  # Self-regulation effect (fixed at 0.5)
        
        # Combined regulatory effect
        total_regulatory_effect = TF_effect + Target_effect
        
        # Effective k_on for Target
        k_Target_eff = k_on_Target * (1 + total_regulatory_effect)
        b_Target = k_Target_eff / (k_Target_eff + k_off_Target)
        
        # Target steady states
        Target_u_ss = (Target_transcription_rate * b_Target) / (Target_mRNA_degradation_rate + splicing_rate)
        Target_s_ss = (Target_u_ss * splicing_rate) / Target_mRNA_degradation_rate
        Target_p_predicted = (Target_s_ss * Target_protein_production_rate) / Target_protein_degradation_rate
        
        # Return equations (should be zero at steady state)
        return [
            TF_p_predicted - TF_p_ss,        # TF equation
            Target_p_predicted - Target_p_ss  # Target equation
        ]
    
    # Solve the coupled system
    # Initial guess: simple steady states without regulation
    TF_initial = (TF_transcription_rate * k_on_TF / (k_on_TF + k_off_TF) * 
                  splicing_rate / (TF_mRNA_degradation_rate + splicing_rate) / 
                  TF_mRNA_degradation_rate * TF_protein_production_rate / TF_protein_degradation_rate)
    
    Target_initial = (Target_transcription_rate * k_on_Target / (k_on_Target + k_off_Target) * 
                      splicing_rate / (Target_mRNA_degradation_rate + splicing_rate) / 
                      Target_mRNA_degradation_rate * Target_protein_production_rate / Target_protein_degradation_rate)
    
    initial_guess = [TF_initial, Target_initial]
    
    try:
        solution = fsolve(steady_state_equations, initial_guess)
        TF_p_ss, Target_p_ss = solution
        
        # Ensure positive values
        TF_p_ss = max(TF_p_ss, 0.1)
        Target_p_ss = max(Target_p_ss, 0.1)
        
    except Exception as e:
        print(f"Failed to solve steady state: {e}")
        TF_p_ss, Target_p_ss = initial_guess
    
    return {
        'average_protein_TF': TF_p_ss,
        'average_protein_Target': Target_p_ss,
    }

def simulate_self_regulation(**sim_args):
    random.seed()
    constants = sim_args.copy()
    # all rates are in hours
    constants.update({
        'TF_mRNA_degradation_rate': np.log(2) / sim_args['mrna_half_life_TF'],
        'Target_mRNA_degradation_rate': np.log(2) / sim_args['mrna_half_life_Target'],
        'TF_transcription_rate': sim_args['burst_size_TF'] * sim_args['k_off_TF'],
        'Target_transcription_rate': sim_args['burst_size_Target'] * sim_args['k_off_Target'],
        'TF_protein_degradation_rate': np.log(2) / sim_args['TF_protein_half_life'],
        'Target_protein_degradation_rate': np.log(2) / sim_args['Target_protein_half_life'],
        'splicing_rate': np.log(2) / (sim_args['splicing_half_life_minutes'] / 60),
        'in_pulse': 0, 'n': sim_args['n'],
        't': 1/60,
    })
    for k, v in constants.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.integer):
            constants[k] = v.astype(float)

    constants['TF_Target_link_EC50'] = 1
    dynamics = 'Michaelis-Menton'
    pulse_time = sim_args['pulse_time']
    num_cells = sim_args['num_cells']

    # trim unused settings
    for unused in {'mrna_half_life_TF', 'mrna_half_life_Target', 'TF_protein_half_life',
                    'Target_protein_half_life', 'splicing_half_life_minutes', 'dynamics', 
                    'pulse_time', 'num_cells', 'burst_size_TF', 'burst_size_Target'}:
        del constants[unused]

    # Find TF mean through calculation
    # averages = get_average_numbers_self_regulation(**constants)
    averages = get_average_numbers_self_loop_regulation(**constants)
    average_protein_TF = averages['average_protein_TF']
    average_protein_Target = averages['average_protein_Target']

    #TF regulates both itself and target
    #TF and Target regulate each other
    constants['TF_Target_link_EC50'] = MM_TF_link(average_protein_TF, sim_args['n'], 16)
    constants['Target_TF_link_EC50'] = MM_TF_link(average_protein_Target, sim_args['n'], 16) 
    
    # burn in TF concentrations for variance
    print("burning in")
    TF_protein_mean, TF_protein_std, Target_protein_mean, Target_protein_std = burn_in_TF_concentration_self_regulation(constants, mean_TF_protein=average_protein_TF, mean_Target_protein=average_protein_Target)
    print("simulating")

    state = generate_random_cells_both_TF(TF_protein_mean, Target_protein_mean, num_cells=num_cells)
    initial_TF = state['TF_protein_1K']
    initial_Target = state['Target_protein_1K']

    samples = {}
    samplesReplicate = {}
    days = 13 #to allow for 1 day after pulse start during which we can sample
    
    # when does pulse happen and for how long?
    day_to_start_pulse_at = 12
    hours_to_track_for = 24
    pulse_start = day_to_start_pulse_at * 60 * hours_to_track_for
    # pulse_start = 0
    pulse_end = int(pulse_start + pulse_time)
    twin_divide_time = 12*60*24 #in minutes - start after reaching steady state
    assert pulse_end < days * 24 * 60
    
    for steps in range(60 * 24 * days + 1):
        if (pulse_start < steps < pulse_end):
            constants['in_pulse'] = 1
        else:
            constants['in_pulse'] = 0
        if (steps == twin_divide_time):
            replicateState = state.copy()
        state = get_new_state_self_regulation(**state, **constants)

        for k in state:
            if k != 'TF_Target_link_EC50' and k != 'Target_TF_link_EC50':
                assert np.all(state[k] >= -0), k
        if steps >= twin_divide_time:
            replicateState = get_new_state_self_regulation(**replicateState, **constants)
            for k in replicateState:
                if k != 'TF_Target_link_EC50' and k != 'Target_TF_link_EC50':
                    assert np.all(state[k] >= -0), k
                
        time_since_start = steps - pulse_start #to sample from start of pulse instead of end
        sampling_interval = 60 #in minutes
        if ((time_since_start >= 0) and (time_since_start % sampling_interval == 0)):
            samples[time_since_start] = state
            samplesReplicate[time_since_start] = replicateState
          
    # convert samples to dataframes
    for k in samples:
        samples[k] = pd.DataFrame(samples[k])
        samples[k]['sampling_time'] = k
    
    for k in samplesReplicate:
        samplesReplicate[k] = pd.DataFrame(samplesReplicate[k])
        samplesReplicate[k]['sampling_time'] = k

    # prepare TF calibration averages for reporting
    TF_values = {k.replace('average_', 'mean TF '): "{:.2f}".format(v) for k, v in averages.items()}

    # prepare constants for reporting
    del constants['in_pulse']
    del constants['TF_Target_link_EC50']
    constants['dynamics'] = dynamics
    constants['pulse time (minutes)'] = pulse_time

    all_samples = pd.concat(samples, ignore_index=True)
    all_samples['replicate'] = 0
    all_samplesReplicate = pd.concat(samplesReplicate, ignore_index=True)
    all_samplesReplicate['replicate'] = 1
    all_samples_both_replicates = pd.concat([all_samples, all_samplesReplicate], ignore_index=True)
    return all_samples_both_replicates


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# @jit(parallel=True, cache=True)
def degrade_poisson(rate, value, step):
    try:
        lam = rate * value * step
        lam = np.clip(lam, 0, None)  # Ensure lam is non-negative
        valueToDegrade = np.minimum(value, np.random.poisson(lam)) #This step is to ensure that you do not degrade more than available molecules
        valueToDegrade = np.maximum(valueToDegrade, 0)
    except:
        print(f"Error in degrade_poisson: rate = {rate}, value = {value}, step = {step}")
        lam = rate * value * step
        invalid = (lam < 0) | ~np.isfinite(lam)
        if np.any(invalid):
            print("Invalid λ values at indices:", np.where(invalid))
            print("Sample invalid λs:", lam[invalid][:5])

        raise
    return valueToDegrade


# @jit(cache=True, parallel=True)
def get_new_state(in_pulse=None,
                  Target_is_bursting=None,
                  TF_is_bursting=None,
                  TF_protein_1K=None,
                  spliced_labeled_Target=None,
                  spliced_labeled_TF=None,
                  spliced_unlabeled_Target=None,
                  spliced_unlabeled_TF=None,
                  unspliced_labeled_Target=None,
                  unspliced_labeled_TF=None,
                  unspliced_unlabeled_Target=None,
                  unspliced_unlabeled_TF=None,
                  unspliced_Target=None,
                  TF_transcription_rate=None,
                  Target_transcription_rate=None,
                  TF_mRNA_degradation_rate=None,
                  Target_mRNA_degradation_rate=None,
                  labeling_efficiency=None,
                  capture_efficiency=None,
                  splicing_rate=None,
                  protein_production_rate=None,
                  protein_degradation_rate=None,
                  k_on_TF=None,
                  k_off_TF=None,
                  k_on_Target=None,
                  k_off_Target=None,
                  TF_Target_link_EC50=None,
                  total_TF_mRNA=None,
                  total_Target_mRNA=None,
                  k_on_Target_adjusted=None,
                  mRNA_ever_produced_Target=None,
                  mRNA_ever_produced_TF=None,
                  protein_ever_produced_TF=None,
                  n=None,
                  t=1 / 60):
    #t = 1/60 since all rates are in hours
    #####################################
    
#   degrade before the continuous calculations
#   degradation of TF mRNA
    unspliced_unlabeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, unspliced_unlabeled_TF, t)
    unspliced_unlabeled_TF = np.maximum(unspliced_unlabeled_TF, 0)
    spliced_unlabeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, spliced_unlabeled_TF, t)
    spliced_unlabeled_TF = np.maximum(spliced_unlabeled_TF, 0)
    
    unspliced_labeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, unspliced_labeled_TF, t)
    unspliced_labeled_TF = np.maximum(unspliced_labeled_TF, 0)
    spliced_labeled_TF -= degrade_poisson(TF_mRNA_degradation_rate, spliced_labeled_TF, t)
    spliced_labeled_TF = np.maximum(spliced_labeled_TF, 0)
   
    # degradation of protein
    TF_protein_1K -= degrade_poisson(protein_degradation_rate, TF_protein_1K * 1000, t) / 1000
    TF_protein_1K = np.maximum(TF_protein_1K, 0)

    # degradation of Target mRNA
    unspliced_unlabeled_Target -= degrade_poisson(Target_mRNA_degradation_rate, unspliced_unlabeled_Target, t)
    unspliced_unlabeled_Target = np.maximum(unspliced_unlabeled_Target, 0)
    spliced_unlabeled_Target -= degrade_poisson(Target_mRNA_degradation_rate, spliced_unlabeled_Target, t)
    spliced_unlabeled_Target = np.maximum(spliced_unlabeled_Target, 0)
    
    unspliced_labeled_Target -= degrade_poisson(TF_mRNA_degradation_rate, unspliced_labeled_Target, t)
    unspliced_labeled_Target = np.maximum(unspliced_labeled_Target, 0)
    spliced_labeled_Target -= degrade_poisson(TF_mRNA_degradation_rate, spliced_labeled_Target, t)
    spliced_labeled_Target = np.maximum(spliced_labeled_Target, 0)
    
    #####################################

    num_cells = TF_protein_1K.shape[0]
    
    ####TF changes####
    
    # absolute new TF mRNA is a function of txn rate and whether TF is currently bursting
    new_TF_mRNA = TF_transcription_rate * TF_is_bursting 
    D_mRNA_ever_produced_TF = new_TF_mRNA.copy()

    # labeling
    # labeling efficiency only matters if in pulse (binary event)
    D_unspliced_labeled_TF = new_TF_mRNA * labeling_efficiency * in_pulse
    D_unspliced_unlabeled_TF = new_TF_mRNA * (1 - labeling_efficiency * in_pulse)

    #note: splicing rate (immature->mature mRNA) is in minutes
    # splicing
    # labeled
    D_unspliced_labeled_TF -= splicing_rate * unspliced_labeled_TF
    D_spliced_labeled_TF = splicing_rate * unspliced_labeled_TF
    # unlabeled
    D_unspliced_unlabeled_TF -= splicing_rate * unspliced_unlabeled_TF
    D_spliced_unlabeled_TF = splicing_rate * unspliced_unlabeled_TF

    # protein
    all_spliced_TF = spliced_unlabeled_TF + spliced_labeled_TF
    D_TF_protein_1K = all_spliced_TF * protein_production_rate
    D_protein_ever_produced_TF = D_TF_protein_1K.copy()

    # switch burst state?
    TF_should_switch_off = TF_is_bursting & (np.random.exponential(1 / k_off_TF, num_cells) < t)
    TF_should_switch_on = (~ TF_is_bursting) & (np.random.exponential(1 / k_on_TF, num_cells) < t)

    ####TF state update####
    #for timestep t (default = 1min)
    TF_protein_1K = TF_protein_1K + D_TF_protein_1K * t
    spliced_labeled_TF = spliced_labeled_TF + D_spliced_labeled_TF * t
    spliced_unlabeled_TF = spliced_unlabeled_TF + D_spliced_unlabeled_TF * t
    unspliced_labeled_TF = unspliced_labeled_TF + D_unspliced_labeled_TF * t
    unspliced_unlabeled_TF = unspliced_unlabeled_TF + D_unspliced_unlabeled_TF * t
    mRNA_ever_produced_TF = mRNA_ever_produced_TF + D_mRNA_ever_produced_TF * t
    protein_ever_produced_TF = protein_ever_produced_TF + D_protein_ever_produced_TF*t
    
    ####Target changes####
    
    # absolute new Target mRNA is a function of txn rate and whether Target is currently bursting
    new_Target_mRNA = Target_transcription_rate * Target_is_bursting
    D_mRNA_ever_produced_Target = new_Target_mRNA.copy()

    # labeling
    # labeling efficiency only matters if in pulse (binary event)
    D_unspliced_labeled_Target = new_Target_mRNA * labeling_efficiency * in_pulse
    D_unspliced_unlabeled_Target = new_Target_mRNA * (1 - labeling_efficiency * in_pulse)

    # splicing
    # labeled
    D_unspliced_labeled_Target -= splicing_rate * unspliced_labeled_Target
    D_spliced_labeled_Target = splicing_rate * unspliced_labeled_Target
    # unlabeled
    D_unspliced_unlabeled_Target -= splicing_rate * unspliced_unlabeled_Target
    D_spliced_unlabeled_Target = splicing_rate * unspliced_unlabeled_Target

    ################################################################################
    ## ----> CHECK THAT LINK FUNCTION IS ON IF NOT DOING STATE-BASED CHECKS <---- ##
    ################################################################################
    # k_on_Target_adjusted = k_on_Target * 1
    k_on_Target_adjusted = k_on_Target*(TF_Target_link_function(TF_protein_1K, TF_Target_link_EC50, n, 16))
    
    Target_should_switch_off = Target_is_bursting & (np.random.exponential(1 / k_off_Target, num_cells) < t)
    Target_should_switch_on = (~ Target_is_bursting) & (np.random.exponential(1 / k_on_Target_adjusted, num_cells) < t)

    ####Target state update####
    #w timestep t (default=1 min)
    spliced_labeled_Target = spliced_labeled_Target + D_spliced_labeled_Target * t
    spliced_unlabeled_Target = spliced_unlabeled_Target + D_spliced_unlabeled_Target * t
    unspliced_labeled_Target = unspliced_labeled_Target + D_unspliced_labeled_Target * t
    unspliced_unlabeled_Target = unspliced_unlabeled_Target + D_unspliced_unlabeled_Target * t
    mRNA_ever_produced_Target = mRNA_ever_produced_Target + D_mRNA_ever_produced_Target * t

    # update whether TF is bursting in this new state
    new_TF_bursting = TF_is_bursting | TF_should_switch_on #It remains on or turned on
    new_TF_bursting[TF_should_switch_off] = False #Turn off whatever was on but should turn off
    TF_is_bursting = new_TF_bursting

    new_Target_bursting = Target_is_bursting | Target_should_switch_on
    new_Target_bursting[Target_should_switch_off] = False
    Target_is_bursting = new_Target_bursting

    total_TF_mRNA = spliced_unlabeled_TF + spliced_labeled_TF + unspliced_unlabeled_TF + unspliced_labeled_TF
    total_Target_mRNA = unspliced_unlabeled_Target + spliced_unlabeled_Target + unspliced_labeled_Target + spliced_labeled_Target
    
    unspliced_TF = unspliced_labeled_TF + unspliced_unlabeled_TF
    unspliced_Target = unspliced_labeled_Target + unspliced_unlabeled_Target

    return {
        "TF_is_bursting": TF_is_bursting,
        "Target_is_bursting": Target_is_bursting,
        "TF_protein_1K": TF_protein_1K,
        "spliced_labeled_Target": spliced_labeled_Target,
        "spliced_labeled_TF": spliced_labeled_TF,
        "spliced_unlabeled_Target": spliced_unlabeled_Target,
        "spliced_unlabeled_TF": spliced_unlabeled_TF,
        "unspliced_labeled_Target": unspliced_labeled_Target,
        "unspliced_labeled_TF": unspliced_labeled_TF,
        "unspliced_unlabeled_Target": unspliced_unlabeled_Target,
        "unspliced_unlabeled_TF": unspliced_unlabeled_TF,
        "unspliced_Target": unspliced_Target,
        "mRNA_ever_produced_Target": mRNA_ever_produced_Target,
        "mRNA_ever_produced_TF": mRNA_ever_produced_TF,
        "protein_ever_produced_TF": protein_ever_produced_TF,
        "k_on_Target_adjusted": k_on_Target_adjusted,
        "total_TF_mRNA": total_TF_mRNA,
        "total_Target_mRNA": total_Target_mRNA}

@njit(fastmath=True)
def TF_Target_link_function(tf_val, EC_50, Hill_coef, max_effect):
        return max_effect * (tf_val**Hill_coef) / ((EC_50**Hill_coef) + (tf_val**Hill_coef))

@njit
def MM_TF_link(TF_mean, Hill_coef, max_effect):
    EC_50 = np.power(max_effect * TF_mean**Hill_coef - TF_mean**Hill_coef, 1.0/Hill_coef)
    # return EC_50
    return TF_mean

#to get TF protein mean for TF protein->Target k_on link function
def get_averages(k_on_TF=None,
                 k_off_TF=None,
                 TF_transcription_rate=None,
                 TF_mRNA_degradation_rate=None,
                 splicing_rate=None,
                 protein_production_rate=None,
                 protein_degradation_rate=None,
                 **other_kwargs):
    average_bursting = k_on_TF / (k_on_TF + k_off_TF)
    average_unspliced_mRNA = TF_transcription_rate * average_bursting / (TF_mRNA_degradation_rate + splicing_rate)
    average_spliced_mRNA = average_unspliced_mRNA * splicing_rate / TF_mRNA_degradation_rate
    average_protein = average_spliced_mRNA * protein_production_rate / protein_degradation_rate
    return dict(average_spliced_mRNA=average_spliced_mRNA, average_protein=average_protein)

def generate_random_cells(mean_TF_protein, num_cells):
    state = {k: np.zeros(int(num_cells)).astype(np.bool if 'is_bursting' in k else np.float64) for k in state_vars}
    # seed TF concentration with a random value - use to check for mixing
    state['TF_protein_1K'] = np.random.uniform(0, 3 * mean_TF_protein, int(num_cells))
    return state

def generate_replicate_cells(num_cells):
    state = {k: np.zeros(int(num_cells)).astype(np.bool if 'is_bursting' in k else np.float64) for k in state_vars}
    return state

#to ensure cells are in steady-state
def burn_in_TF_concentration(constants, num_cells=500, mean_TF_protein=20, days=12):
    state = generate_random_cells(mean_TF_protein, num_cells=num_cells)
    for day in range(days):
        for steps in range(60 * 24): #This means that burn in time is 6 days - adjust accordingly
            state = get_new_state(**state, **constants)
    return state['TF_protein_1K'].mean(), state['TF_protein_1K'].std()

#Simulation function for synchronous division
def simulate(**sim_args):
    random.seed()
    constants = sim_args.copy()
    # all rates are in hours
    constants.update({
        'TF_mRNA_degradation_rate': np.log(2) / sim_args['mrna_half_life_TF'],
        'Target_mRNA_degradation_rate': np.log(2) / sim_args['mrna_half_life_Target'],
        'TF_transcription_rate': sim_args['burst_size_TF'] * sim_args['k_off_TF'],
        'Target_transcription_rate': sim_args['burst_size_Target'] * sim_args['k_off_Target'],
        'TF_protein_degradation_rate': np.log(2) / sim_args['TF_protein_half_life'],
        'Target_protein_degradation_rate': np.log(2) / sim_args['Target_protein_half_life'],
        'splicing_rate': np.log(2) / (sim_args['splicing_half_life_minutes'] / 60),
        'in_pulse': 0, 'n': sim_args['n'],
        't': 1/60,
    })
    for k, v in constants.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.integer):
            constants[k] = v.astype(float)


    constants['TF_Target_link_EC50'] = 1
    dynamics = 'Michaelis-Menton'
    pulse_time = sim_args['pulse_time']
    num_cells = sim_args['num_cells']

    # trim unused settings
    for unused in {'mrna_half_life_TF', 'mrna_half_life_Target', 'TF_protein_half_life',
                    'Target_protein_half_life', 'splicing_half_life_minutes', 'dynamics', 
                    'pulse_time', 'num_cells', 'burst_size_TF', 'burst_size_Target'}:
        del constants[unused]

    # Find TF mean through calculation
    averages = get_averages(**constants)
    average_protein_TF = averages['average_protein_TF']
    average_protein_Target = averages['average_protein_Target']

    # burn in TF concentrations for variance
    print("burning in")
    TF_protein_mean, TF_protein_std, Target_protein_mean, Target_protein_std = burn_in_TF_concentration(constants, mean_TF_protein=average_protein_TF, mean_Target_protein=average_protein_Target)
    print("simulating")

    state = generate_random_cells(TF_protein_mean, num_cells=num_cells)
    initial_TF = state['TF_protein_1K']

    constants['TF_Target_link_EC50'] = MM_TF_link(TF_protein_mean, sim_args['n'], 16)
    constants['Target_TF_link_EC50'] = MM_TF_link(Target_protein_mean, sim_args['n'], 16)
    
    samples = {}
    samplesReplicate = {}
    days = 13 #to allow for 1 day after pulse start during which we can sample
    
    # when does pulse happen and for how long?
    day_to_start_pulse_at = 12
    hours_to_track_for = 24
    pulse_start = day_to_start_pulse_at * 60 * hours_to_track_for
    # pulse_start = 0
    pulse_end = int(pulse_start + pulse_time)
    twin_divide_time = 12*60*24 #in minutes - start after reaching steady state
    assert pulse_end < days * 24 * 60
    
    for steps in range(60 * 24 * days + 1):
        if (pulse_start < steps < pulse_end):
            constants['in_pulse'] = 1
        else:
            constants['in_pulse'] = 0
        if (steps == twin_divide_time):
            replicateState = state.copy()
        state = get_new_state(**state, **constants)

        for k in state:
            if k != 'TF_Target_link_EC50':
                assert np.all(state[k] >= -0), k
        if steps >= twin_divide_time:
            replicateState = get_new_state(**replicateState, **constants)
            for k in replicateState:
                if k != 'TF_Target_link_EC50':
                    assert np.all(state[k] >= -0), k
                
        time_since_start = steps - pulse_start #to sample from start of pulse instead of end
        sampling_interval = 60 #in minutes
        if ((time_since_start >= 0) and (time_since_start % sampling_interval == 0)):
            samples[time_since_start] = state
            samplesReplicate[time_since_start] = replicateState
          
    # convert samples to dataframes
    for k in samples:
        samples[k] = pd.DataFrame(samples[k])
        samples[k]['sampling_time'] = k
    
    for k in samplesReplicate:
        samplesReplicate[k] = pd.DataFrame(samplesReplicate[k])
        samplesReplicate[k]['sampling_time'] = k

    # prepare TF calibration averages for reporting
    TF_values = {k.replace('average_', 'mean TF '): "{:.2f}".format(v) for k, v in averages.items()}

    # prepare constants for reporting
    del constants['in_pulse']
    del constants['TF_Target_link_EC50']
    constants['dynamics'] = dynamics
    constants['pulse time (minutes)'] = pulse_time

    all_samples = pd.concat(samples, ignore_index=True)
    all_samples['replicate'] = 0
    all_samplesReplicate = pd.concat(samplesReplicate, ignore_index=True)
    all_samplesReplicate['replicate'] = 1
    all_samples_both_replicates = pd.concat([all_samples, all_samplesReplicate], ignore_index=True)
    return all_samples_both_replicates

#####################################################################################################################################
#Simulation for asynchronous division
#####################################################################################################################################

def saveTwinDataToCsv(samples, samplesReplicate, twinPairs):
    dfs_orig = []
    dfs_repl = []
    reverseTwinPairs = {v: k for k, v in twinPairs.items()}
    for t, state in samples.items():
        df_t = pd.DataFrame(state)
        df_t["sampling_time"] = t
        df_t["cell_index"] = list(range(len(df_t)))
        df_t["replicate"] = 0
        df_t["replicate_index"] = df_t["cell_index"].map(twinPairs)
        dfs_orig.append(df_t)

    for t, state in samplesReplicate.items():
        df_r = pd.DataFrame(state)
        df_r["sampling_time"] = t
        df_r["replicate"] = 1
        df_r["replicate_index"] = list(range(len(df_r)))
        df_r["cell_index"] = df_r["replicate_index"].map(reverseTwinPairs)
        dfs_repl.append(df_r)

    df_orig_all = pd.concat(dfs_orig, ignore_index=True)
    df_repl_all = pd.concat(dfs_repl, ignore_index=True)
    return pd.concat([df_orig_all, df_repl_all], ignore_index=True)

def generateTwinDivideTime(num_cells, days, modeDay = 12.3, sigma = 0.005):
    """
    Generate twin divide time from log-normal distribution with mode at day 13
    Ensures all cells divide before simulation ends
    """
    # For log-normal distribution: mode = exp(mu - sigma^2)
    # Therefore: mu = ln(mode) + sigma^2
    mu = np.log(modeDay) + sigma**2
    
    # Keep sampling until all cells divide within simulation time
    maxDivideDay = days - 1  # Leave 1 day buffer
    twinDivideTimes = np.zeros(num_cells, dtype=float)

    
    for i in range(num_cells):
        while True:
            divideDay = np.random.lognormal(mu, sigma)
            if divideDay <= maxDivideDay:
                twinDivideTimes[i] = divideDay
                break
    #First save twin divide times in days
    df = pd.DataFrame(twinDivideTimes, columns=['twin_divide_time_days'])
    # Save to CSV
    output_path = f'/home/mzo5929/Keerthana/grnInference/simulationData/asynchronous_division/twin_divide_times_sigma_{sigma:.3f}.csv'
    df.to_csv(output_path, index=True, index_label='cell_index')
    #Plot histogram 
    import matplotlib.pyplot as plt
    plt.hist(twinDivideTimes, bins=30, edgecolor='black')
    plt.xlabel('Twin Divide Time (days)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Twin Divide Times')
    plt.savefig(f'/home/mzo5929/Keerthana/grnInference/simulationData/asynchronous_division/twin_divide_times_histogram_sigma_{sigma:.3f}.png')
    twinDivideTimes = np.round(twinDivideTimes * 24 * 60).astype(int)  # Convert to minutes
    return twinDivideTimes

def simulate_asynchronous(**sim_args):
    print("Simulating with asynchronous division")
    random.seed()
    constants = sim_args.copy()
    # all rates are in hours
    sigma = sim_args['sigma']
    constants.update({
        'TF_mRNA_degradation_rate': np.log(2) / sim_args['mrna_half_life_TF'],
        'Target_mRNA_degradation_rate': np.log(2) / sim_args['mrna_half_life_Target'],
        'TF_transcription_rate': sim_args['burst_size_TF'] * sim_args['k_off_TF'],
        'Target_transcription_rate': sim_args['burst_size_Target'] * sim_args['k_off_Target'],
        'protein_degradation_rate': np.log(2) / sim_args['protein_half_life'],
        'splicing_rate': np.log(2) / (sim_args['splicing_half_life_minutes'] / 60),
        'in_pulse': 0,
        't': 1/60,
    })
    for k, v in constants.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.integer):
            constants[k] = v.astype(float)


    constants['TF_Target_link_EC50'] = 1
    dynamics = 'Michaelis-Menton'
    pulse_time = sim_args['pulse_time']
    num_cells = sim_args['num_cells']
    ###############################################################################################################
    #Make twin divide time a distribution such that the mode is 12.8 days and it is lognormal distributed
    twinDivideTimes = generateTwinDivideTime(num_cells, 14, modeDay=12.8, sigma=constants['sigma'])

    # Track which cells have divided
    cellsHaveDivided = np.zeros(num_cells, dtype=bool)

    # Track twin relationships - maps original cell index to replicate cell index
    twinPairs = {}  # {original_cell_idx: replicate_cell_idx}
    nextReplicateIdx = 0  # Counter for assigning replicate indices

    # Initialize replicate state arrays before the simulation loop
    replicateState = generate_replicate_cells(num_cells=num_cells)
    for k in replicateState:
        assert np.all(replicateState[k] == 0), k  # Add this line to confirm
    ###############################################################################################################
    # trim unused settings
    for unused in {'mrna_half_life_TF', 'mrna_half_life_Target', 'protein_half_life',
                   'splicing_half_life_minutes', 'dynamics', 'pulse_time', 'num_cells',
                   'burst_size_TF', 'burst_size_Target', 'sigma'}:
        del constants[unused]

    # Find TF mean through calculation
    averages = get_averages(**constants)
    average_protein = averages['average_protein']

    # burn in TF concentrations for variance
    print("burning in")
    TF_protein_mean, TF_protein_std = burn_in_TF_concentration(constants, mean_TF_protein=average_protein)
    print("simulating")

    state = generate_random_cells(TF_protein_mean, num_cells=num_cells)
    initial_TF = state['TF_protein_1K']

    constants['TF_Target_link_EC50'] = MM_TF_link(TF_protein_mean, 2, 16)
    
    samples = {}
    samplesReplicate = {}
    days = 14 #to allow for 2 day after pulse start during which we can sample
    
    # when does pulse happen and for how long?
    day_to_start_pulse_at = 12
    hours_to_track_for = 48
    pulse_start = day_to_start_pulse_at * 60 * 24
    # pulse_start = 0
    pulse_end = int(pulse_start + pulse_time)

    ###################################################
    division_time = {}
    for steps in range(60 * 24 * days + 1):
        if (pulse_start < steps < pulse_end):
            constants['in_pulse'] = 1
        else:
            constants['in_pulse'] = 0
        
        # Check which cells are dividing at this time step
        cellsDividingNow = (twinDivideTimes == steps) & (~cellsHaveDivided)
        
        if np.any(cellsDividingNow):
            # For each cell dividing now, assign it a replicate index and record the twin pair
            dividingCellIndices = np.where(cellsDividingNow)[0]
            for originalIdx in dividingCellIndices:
                division_time[originalIdx] = steps - pulse_start  # Record division time relative to pulse start
                replicateIdx = nextReplicateIdx
                twinPairs[originalIdx] = originalIdx

                # Copy state of dividing cell to its replicate
                for k in state:
                    replicateState[k][originalIdx] = state[k][originalIdx]
                
                nextReplicateIdx += 1
            
            # Mark these cells as having divided
            cellsHaveDivided[cellsDividingNow] = True
        state = get_new_state(**state, **constants)

        for k in state:
            if k != 'tfTargetLinkEc50':
                assert np.all(state[k] >= -0), k
        
        # Update replicate states for cells that have already divided
        if twinPairs:
            validIdx = list(twinPairs.values())
            if validIdx:  # Ensure validIdx is not empty
                # Step 1: Copy only the relevant replicate values
                replicateStateSubset = {k: np.copy(replicateState[k][validIdx]) for k in replicateState}

                # Step 2: Update only those values
                updatedSubset = get_new_state(**replicateStateSubset, **constants)

                # Step 3: Overwrite only the corresponding indices in the full replicateState
                for k in replicateState:
                    replicateState[k][validIdx] = updatedSubset[k]

        time_since_start = steps - pulse_start #to sample from start of pulse instead of end
        sampling_interval = 60#in minutes
        if ((time_since_start >= 0) and (time_since_start % sampling_interval == 0)):
            samples[time_since_start] =  {
                k: state[k].copy() 
                for k in state
            }
            uninitializedIdx = list(set(range(num_cells)) - set(twinPairs.values()))
            for k in replicateState:
                assert np.all(replicateState[k][uninitializedIdx] == 0), f"Unexpected nonzero in {k}"

            # Add replicate state to samplesReplicate
            samplesReplicate[time_since_start] = {
                k: replicateState[k].copy() 
                for k in replicateState
            }
            #Save at every time point
            # if (time_since_start % (24*60) == 0):
            #     df_time_point_replicate = pd.DataFrame(replicateState)
            #     df_time_point_replicate['sampling_time'] = time_since_start
            #     df_time_point_replicate.to_csv(f"/home/mzo5929/Keerthana/grnInference/simulationData/asynchronous_division/replicate_state_at_time_{time_since_start}.csv")
    # convert samples to dataframes
    for k in samples:
        samples[k] = pd.DataFrame(samples[k])
        samples[k]['cell_index'] = samples[k].index
        samples[k]['sampling_time'] = k
    
    for k in samplesReplicate:
        samplesReplicate[k] = pd.DataFrame(samplesReplicate[k])
        samplesReplicate[k]['cell_index'] = samplesReplicate[k].index
        samplesReplicate[k]['sampling_time'] = k

    # prepare TF calibration averages for reporting
    TF_values = {k.replace('average_', 'mean TF '): "{:.2f}".format(v) for k, v in averages.items()}

    # prepare constants for reporting
    del constants['in_pulse']
    del constants['TF_Target_link_EC50']
    constants['dynamics'] = dynamics
    constants['pulse time (minutes)'] = pulse_time
    # all_samples_both_replicates = saveTwinDataToCsv(samples, samplesReplicate, twinPairs)
    # print(all_samples_both_replicates.columns)
    all_samples = pd.concat(samples, ignore_index=True)
    all_samples['replicate'] = 0
    all_samplesReplicate = pd.concat(samplesReplicate, ignore_index=True)
    all_samplesReplicate['replicate'] = 1
    all_samples_both_replicates = pd.concat([all_samples, all_samplesReplicate], ignore_index=True)

    #save division time
    division_time_df = pd.DataFrame(list(division_time.items()), columns=['cell_index', 'division_time'])
    division_time_df = division_time_df.sort_values(by='cell_index').reset_index(drop=True)

    division_time_df.to_csv(f'/home/mzo5929/Keerthana/grnInference/simulationData/asynchronous_division/division_time_variation_{sigma:.3f}.csv', index=False)
    return all_samples_both_replicates
    