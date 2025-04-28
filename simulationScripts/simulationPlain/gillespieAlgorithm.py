#This contains the scripts for gillespie algorithm implementation

import numpy as np
import multiprocess
import numba
from numpy.random import default_rng
import tqdm
from concurrent.futures import ProcessPoolExecutor as ExecutorPool
Pool = ExecutorPool

#Function for multithreading
def _gillespieMultithreadFunc(args):
    logQueue, seed, newArgs = args
    # logQueue.put("entered _gillespieMultithreadFunc and clear")
    rng = default_rng(seed)
    return _gillespieSSA(*newArgs, randomGenerator = rng)

@numba.njit(cache=True)
def _copyPopulation(populationPrevious, population, nSpecies):
    for i in range(nSpecies):
        populationPrevious[i] = population[i]

#Function to draw a reaction and time taken to do that reaction
# @numba.njit(cache=True)
# def _draw(propensities, population, t, updatePropensity, randomGenerator):
#     #Compute propensities
#     updatePropensity(propensities, population, t)
    
#     #Sum of propensities to find the mean of exponential distribution of time taken for event and calculate probabilty of each event

#     propensitiesSum = np.sum(propensities)

#     #End the simulation if propensitiesSum is zero
#     if propensitiesSum == 0.0:
#         return -1, -1.0
    
#     time = randomGenerator.exponential(1.0/propensitiesSum)

#     #Draw reaction given the propensities
#     #UPDATED
#     cumulativeSum = np.cumsum(propensities)
#     reaction = np.searchsorted(cumulativeSum, randomGenerator.random() * propensitiesSum)
#     # reaction = _sampleDiscrete(propensities, propensitiesSum, randomGenerator)
#     return reaction, time

@numba.njit(fastmath=True)
def _draw(propensities, population, t, updatePropensity, randomGenerator):
    #Compute propensities
    updatePropensity(propensities, population, t)
    
    #Sum of propensities to find the mean of exponential distribution of time taken for event and calculate probabilty of each event

    propensitiesSum = np.sum(propensities)

    #End the simulation if propensitiesSum is zero
    if propensitiesSum == 0.0:
        return -1, -1.0
    
    time = randomGenerator.exponential(1.0/propensitiesSum)

    #Draw reaction given the propensities
    #UPDATED
    cumulativeSum = 0.0
    reaction =  randomGenerator.random() * propensitiesSum
    for i in range(len(propensities)):
        cumulativeSum += propensities[i]
        if cumulativeSum > reaction:
            return i, time

    # reaction = _sampleDiscrete(propensities, propensitiesSum, randomGenerator)
    return -1, -1

@numba.njit(fastmath=True)
def _sampleDiscrete(propensities, propensitiesSum, randomGenerator):
    selection = randomGenerator.random() * propensitiesSum
    i = 0
    pSum = 0
    while(pSum < selection):
        pSum += propensities[i]
        if(pSum > selection):
            return i
        i+=1
    return -1

@numba.njit( fastmath=True)
def _traj(nSpecies, updatePropensity, populationInit, parameterUpdateValues, timePoints, randomGenerator = None):
    #Initialise output
    populationOutputCell = np.empty((len(timePoints), parameterUpdateValues.shape[1]), dtype=np.int64)
    #Initialising the system for simulation
    jTime = 1
    j = 0
    t = timePoints[0]
    # population = populationInit.copy()
    # populationPrevious = population.copy()
    population = np.empty_like(populationInit)
    populationPrevious = np.empty_like(population)
    for i in range(len(populationInit)):
        population[i] = populationInit[i]
        populationPrevious[i] = populationInit[i]
    population = populationInit.copy()
    populationOutputCell[0, :] = population
    propensities = np.zeros(parameterUpdateValues.shape[0])
    while(j < len(timePoints)):
        while t < timePoints[jTime]:
            #drawing the event and time for it
            event, dt = _draw(propensities, population, t, updatePropensity, randomGenerator = randomGenerator)
            if event == -1:
                #skip to the end of population because sum of propensities is 0
                t = timePoints[-1]
            else:
                #update population
                _copyPopulation(populationPrevious, population, nSpecies)
                population += parameterUpdateValues[event, :]
                #increment time
                t += dt

        #Update the index in timePoints to be chosen such that it is the closest after this event\
        j = np.searchsorted((timePoints > t).astype(np.int64), 1)
        #Update the population in the output population
        for k in np.arange(jTime, min(j, len(timePoints))):
            populationOutputCell[k:] = populationPrevious
            
        #Increment index to current index
        jTime = j
    return populationOutputCell
# @numba.njit(cache = True)  
# def _traj(nSpecies, updatePropensity, populationInit, parameterUpdateValues,
#            timePoints, randomGenerator=None):
#     # Initialize output
#     populationOutputCell = np.empty((len(timePoints), parameterUpdateValues.shape[1]), dtype=np.int64)
#     population = populationInit.copy()
#     propensities = np.zeros(parameterUpdateValues.shape[0])

#     t = timePoints[0]
#     populationOutputCell[0, :] = population
#     jTime = 1
#     j = 0
#     loopNumber = 0
#     while j < len(timePoints) - 1:
#         loopNumber = loopNumber + 1
#         if loopNumber > 100000 and loopNumber > len(timePoints):
#             print("Infinite Loop")
#             return
#         while t < timePoints[jTime]:
#             event, dt = _draw(propensities, population, t, updatePropensity, randomGenerator=randomGenerator)
#             if event == -1:
#                 t = timePoints[-1]
#                 break
#             population += parameterUpdateValues[event, :]
#             t += dt

#         j = np.searchsorted(timePoints, t, side='right')
#         if j >= len(timePoints):
#             j = len(timePoints) - 1

#         populationOutputCell[jTime:min(j, len(timePoints)), :] = population
#         jTime = j
#     return populationOutputCell

def _gillespieSSA(
        updatePropensity,
        parameterUpdateValues,
        populationInit,
        timePoints,
        simPerThread = 1, #simPerThread is number of cells we are simulating
        progressBar = False,
        logQueue = None,
        randomGenerator = None

):
    #Function to run the Gillespie algorithm
    #number of species
    # logQueue.put(f"Starting _gillespie with {simPerThread} simulations/cells.")
    nSpecies = parameterUpdateValues.shape[1]

    #Ensuring the number of species is equal to the number of columns in population (i.e. the number of species we are tracking)
    if nSpecies != len(populationInit):
        raise(ValueError("Number of rows in update parameters must be equal to the length of populationInit"))
    
    #Initialise output
    populationOutput = [np.empty((len(timePoints), parameterUpdateValues.shape[1]), dtype=np.int64) for _ in range(simPerThread)]
    #Progress bar
    iterator = range(simPerThread)
    if progressBar == "notebook":
            iterator = tqdm.notebook.tqdm(range(simPerThread))
    elif progressBar:
        iterator = tqdm.tqdm(range(simPerThread))
    # (f"Starting the _traj for {simPerThread} iterations")
    #Calculating the population over time
    for i in iterator:
        populationOutput[i] = _traj(nSpecies, updatePropensity, populationInit, parameterUpdateValues, timePoints, randomGenerator = randomGenerator)
    ###PRINT
    # logQueue.put(f"Finished the _traj for {simPerThread} iterations.")
    # #logQueue.put(f"Dimension details of populationOutput:\nTrajectories: {len(populationOutput)}\tTime points: {len(populationOutput[0])}\tSpecies: {len(populationOutput[0][0])}")
    # print("Dimension details of populationOutput:", "Trajectories:", len(populationOutput), "Time points:", len(populationOutput[0]), "Species:", len(populationOutput[0][0]))
    # logQueue.put(f"Finished _gillespie with {simPerThread} simulations/cells.")
    return populationOutput



def gillespieSSA(
    updatePropensity,
    parameterUpdateValues,
    populationInit,
    timePoints,
    baseSeed,
    simPerThread, #can be simulation or cell
    numThreads = 1,
    progressBar = False,
    logQueue = None
):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from probability distribution of particle counts over time.

    Parameters
    ----------
    updatePropensity : function
        Function with call signature
        `updatePropensity(propensities, population, t) that takes
        the current propensities and population of particle counts and
        updates the propensities for each reaction. It does not return
        anything.
    parameterUpdateValues : ndarray, shape (numReactions, numChemicalSpecies)
        Entry i, j gives the change in particle counts of species j
        for chemical reaction i.
    populationInit: array-like, shape (numChemicalSpecies)
        Array of initial populations of all chemical species.
    timePoints : array-like, shape (numTimePoints,)
        Array of points in time for which to sample the probability
        distribution.
    simPerThread : int, default 1
        Number of trajectories to sample per thread.
    numThreads : int, default 1
        Number of threads to use in the calculation.
    progressBar : str or bool, default False
        If True, use standard tqdm progress bar. If 'notebook', use
        tqdm.notebook progress bar. If False, no progress bar.

    Returns
    -------
    samples : ndarray
        Entry i, j, k is the count of chemical species k at time
        time_points[j] for trajectory of i th cell. The shape of the array is
        (simPerThread*n_threads(nCells/number of simulations), num_time_points, num_chemical_species).

    """
    #Check inputs and reformat it as neecssary
    try:
        populationInit = populationInit.astype(int)
    except:
        logQueue.put("line - 1")
    try:
        parameterUpdateValues = parameterUpdateValues.astype(int)
    except:
        logQueue.put("line - 2")
    try:
        timePoints = np.array(timePoints, dtype=float)
    except:
        logQueue.put("line - 3")
    if numThreads == 1:
        rng = default_rng(baseSeed)
        # seeds = rng.integers(0, 2**32, numThreads, dtype=np.uint32)

        # threadArgs = [(seeds[i], inputArgs) for i in range(numThreads)]
        # logQueue.put(f"Starting _gillespieSSA with one thread and {simPerThread} simulations/cells. ")
        populationOutput  = _gillespieSSA(
              updatePropensity,
              parameterUpdateValues,
              populationInit,
              timePoints,
              simPerThread=simPerThread,
              progressBar=False,
              randomGenerator=rng,
              logQueue = logQueue
        )
        if len(populationOutput) ==1:
            return populationOutput[0].reshape((1, *populationOutput[0].shape))
        else:
            return np.stack(populationOutput, axis=0)
    else:
        # logQueue.put(f"Starting _gillespieMultithreadFunc with {numThreads} threads and {simPerThread} simulations/cells. ")
        #creating an array to feed it into function separately for each thread
        inputArgs = (
             updatePropensity,
              parameterUpdateValues,
              populationInit,
              timePoints,
              simPerThread,
              progressBar,
              logQueue
        )
        #TODO - DONE with base seed, generate a seed for each thread and input it
        rng = default_rng(baseSeed)
        seeds = rng.integers(0, 2**32, numThreads, dtype=np.uint32)
        threadArgs = [(logQueue, seeds[i], inputArgs) for i in range(numThreads)]

        # Execute the multithreaded simulation
        with Pool(max_workers=numThreads) as p:
            results = list(p.map(_gillespieMultithreadFunc, threadArgs))
        # print(f"np.array(results).shape = {np.array(results).shape}")
        # print("results[0]")
        # print("Shape of results[0]:", np.array(results[0]).shape)
        # print(f"SimulationPerThread = {simPerThread}")
        # print(f"numThreads = {numThreads}")

        populationOutputs = [results[i][k] for i in range(numThreads) for k in range(simPerThread)]
        # print("Final combined shape of populationOutputs", np.array(populationOutputs).shape)
        if len(populationOutputs) ==1:
            return populationOutputs[0].reshape((1, *populationOutputs[0].shape))
        else:
            return np.stack(populationOutputs, axis=0)
