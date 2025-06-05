import numpy as np
from numpy import random
import pandas as pd
import os.path
from numba import jit, njit

from numba import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)

random.seed()

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
    # k_on_Target_adjusted = k_on_Target*(TF_Target_link_function(TF_protein_1K, TF_Target_link_EC50, n, 16))
    k_on_Target_adjusted = k_on_Target*(1 + TF_Target_link_function(TF_protein_1K, TF_Target_link_EC50, n, 16))
    
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

#to ensure cells are in steady-state
def burn_in_TF_concentration(constants, num_cells=500, mean_TF_protein=20, days=12):
    state = generate_random_cells(mean_TF_protein, num_cells=num_cells)
    for day in range(days):
        for steps in range(60 * 12): #This means that burn in time is 6 days - adjust accordingly
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
        'protein_degradation_rate': np.log(2) / sim_args['protein_half_life'],
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
    for unused in {'mrna_half_life_TF', 'mrna_half_life_Target', 'protein_half_life',
                   'splicing_half_life_minutes', 'dynamics', 'pulse_time', 'num_cells',
                   'burst_size_TF', 'burst_size_Target'}:
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

    constants['TF_Target_link_EC50'] = MM_TF_link(TF_protein_mean, sim_args['n'], 16)
    
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

    