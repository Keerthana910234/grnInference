#This is the script containing helper functions for simulation along with Cell class
import numba
from gillespieAlgorithm import gillespieSSA
import numpy as np
import bokeh
import bokeh.plotting
import matplotlib.pyplot as plt
from bokeh.plotting import figure as bokehFig
from scipy.stats import lognorm
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import nbinom
from scipy.optimize import minimize

# @numba.njit
# def setSeed(seedValue):
#     np.random.seed(seedValue)
#     return

def setupGillespieParams(initStates, reactions, propensityParameters, logQueue = None):
    """
    Set up the parameters (initial population, changes in population for each reaction, function to change propensities based off updated population) for Gillespie's algorithm based on the initial states, reactions, and propensity parameters.

    Parameters
    ----------
    initStates : array-like, shape(numChemicalSpecies)
        It contains the initial values for all the species. All genes are set to be in off state and mRNA content is 0.
    reactions : ndarray, shape (numReactions, 6)
        It contains information about the species involved in each reaction and the changes in their population if that reaction occurs along with the propensity of that reaction to occur.
    propensityParameters: 2D-array, shape (8, 2)
        A table containing the values for all reaction parameters to be substituted in the propensity expressions

    Returns
    -------
    populationInit : np.ndarray, shape (numChemicalSpecies,)
        The initial population counts for each chemical species.
    parameterUpdateValues : np.ndarray, shape (numReactions, numChemicalSpecies)
        An array that specifies the changes in population for each reaction. Each row corresponds to a reaction, and each column corresponds to a species.
    updateFunc : str
        A string representing the function definition for updating the propensities of the reactions.
    speciesIndex : dict
        A dictionary mapping each chemical species to its index.

    """
    #remove any Nan containing rows
    # logQueue.put("Setting up Gillespie Parameters")
    initStates = initStates.dropna() 
    reactions = reactions.dropna()
    propensityParameters = propensityParameters.dropna()

    #Creating indices for each of chemical species
    speciesList = initStates.drop_duplicates()
    speciesList = speciesList.sort_values(by='species')
    speciesIndex = {species: i for i, species in enumerate(speciesList['species'])}

    #Setting initial counts for all species so that the indices match
    populationInit = np.array(speciesList['count'], dtype=np.int64)

    #Setting up the dictionary for propensity parametrs to their set value (eg. n = 0.5)
    propensityParameters = propensityParameters.drop_duplicates(subset='parameter')
    parameterDictionary = {'{' + row['parameter'] + '}': row['value'] for _, row in propensityParameters.iterrows()}
    parameterUpdateValues = []
    propensityFormulas = []

    #iterating through rows of reactions to create the parameterUpdateValues for each species involved in the reaction

    for i,row in reactions.iterrows():
        if row['species1'] not in speciesIndex:
            raise ValueError(('Species {} not in initial state').format(row['species1']))
        speciesIdx = speciesIndex[row['species1']]

        #Creating update parameters for each species and updating the column is it is not 0
        updateParamRow = [0 for _ in range(len(speciesIndex))]
        updateParamRow[speciesIdx] = row['change1']

        #If species2 exists that needs to be updated too
        if row['species2'] != '-':
            if row['species2'] not in speciesIndex:
                raise(ValueError('Species {} not in initial state').format(row['species2']))
            species2Idx = speciesIndex[row['species2']]
            updateParamRow[species2Idx] = row['change2']

        parameterUpdateValues.append(updateParamRow)

        #Converting the formula for propensity into python code - change species value to its index number and replace parameters with its values
        formula = row['propensity']

        #replacing species with their indices
        for species in speciesIndex:
            formula = formula.replace(species, 'population[{}]'.format(speciesIndex[species]))
        
        #Inserting the numerical parameter values instead of {parameter} notation in the propensity expression
        for key in parameterDictionary.keys():
            formula = formula.replace(key, str(parameterDictionary[key]))
            
        # Adding the time condition to propensity formula if it exists
        if row['time']!= '-':
            propensityFormulas.append('propensities[{}]'.format(i)+ formula + 'if (' + row['time'] + ') else 0')
        else:
            propensityFormulas.append('propensities[{}] = '.format(i) + formula)
        
    parameterUpdateValues = np.array(parameterUpdateValues, dtype=np.int64)
    #Prepare formulas to hardcode into update function
    propensityFormulas = '\n\t'.join(propensityFormulas)

    #defining update function which contains all the propensity formulas to run before the execution of simulation
    updateFunc = '@numba.njit(fastmath = True)\ndef updatePropensity(propensities, population,t):\n\t' + propensityFormulas
    # logQueue.put("Finished setting up Gillespie Parameters")
    return populationInit, parameterUpdateValues, updateFunc, speciesIndex

def plotSim(samples,
            timePoints,
            simPerThread,
            numThreads,
            speciesIndex,
            plotAll=False,
            trendline=True,
            shouldSave=False,
            savePath=None):
    """
    Plot simulation trajectories using matplotlib.
    
    Args:
        samples: Array of shape (numSimulations, numTimePoints, numSpecies)
        timePoints: Array of time points
        simPerThread: Number of simulations per thread
        numThreads: Number of threads used
        speciesIndex: Dictionary mapping species names to indices
        plotAll: If True, plot all trajectories, otherwise sample
        trendline: If True, plot mean trajectory
        shouldSave: If True, save plot to file
        savePath: Path to save plot file
    """

    # Filter species to only include mRNA
    speciesList = [species for species in speciesIndex if 'mRNA' in species]

    # Calculate number of plots and grid layout
    numPlots = len(speciesList)
    numCols = min(3, numPlots)
    numRows = (numPlots + numCols - 1) // numCols

    # Create figure and subplots
    fig, axes = plt.subplots(numRows,
                             numCols,
                             figsize=(5 * numCols, 4 * numRows))
    if numPlots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Calculate divisor for trajectory sampling
    if plotAll:
        divisor = 1
    else:
        divisor = max(1, int(numThreads * simPerThread /
                             50))  # plot only 50 trajectories

    # Set transparency based on trendline
    transparency = 0.3 if trendline else 0.9

    # Create plots for each species
    for idx, species in enumerate(speciesList):
        ax = axes[idx]
        speciesIdx = speciesIndex[species]

        # Plot individual trajectories
        for trajectory in samples[::divisor, :, speciesIdx]:
            ax.plot(timePoints,
                    trajectory,
                    linewidth=0.3,
                    alpha=transparency,
                    color='blue')

        # Plot trendline if requested
        if trendline:
            mean_trajectory = np.mean(samples[:, :, speciesIdx], axis=0)
            ax.plot(timePoints,
                    mean_trajectory,
                    linewidth=1,
                    color='orange',
                    label='Mean')
            ax.legend()

        # Set labels and title
        ax.set_xlabel('Dimensionless Time')
        ax.set_ylabel(f'Number of {species}')
        ax.set_title(species)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

    # Remove any empty subplots
    for idx in range(numPlots, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout
    plt.tight_layout()

    # Save or show plot
    if shouldSave and savePath:
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return

# def plotSim(samples, timePoints, simPerThread, numThreads, speciesIndex, plotAll = False, trendline = True, shouldSave = False, savePath = None):
#     plots = []
#     speciesOrder = []
#     for species in speciesIndex:
#         if 'mRNA' not in species:
#             continue
#         plots.append(
#             bokehFig(
#                 frame_width = 300,
#                 frame_height = 200,
#                 x_axis_label = 'dimensionless time',
#                 y_axis_label = 'number of {}'.format(species)
#             )
#         )
#         speciesOrder.append(species)

#     if plotAll:
#         divisor = 1
#     else:
#         divisor = int(numThreads*simPerThread/50) #plot only 50 trajectories
#         divisor = max(divisor, 1)

#     transparency = 0.3 if trendline else 0.9

#     for (plot,species) in zip(plots, speciesOrder):
#         index = speciesIndex[species]
#         for x in samples[::divisor, :, index]:
#             plot.line(timePoints, x, line_width = 0.3, alpha = transparency, line_join = 'bevel')
#         if trendline:
#             plot.line(
#                 timePoints,
#                 samples[:,:, index].mean(axis = 0),
#                 line_width = 1,
#                 color = 'orange',
#                 line_join = 'bevel'
#             )

#     for plot in plots:
#         plot.x_range = plots[0].x_range
#         plot.y_range = plots[0].y_range
#     if(shouldSave and savePath):
#          bokeh.io.save(plots, filename=savePath, title = "Steady state levels")
#     # bokeh.io.show(bokeh.layouts.gridplot(plots, ncols = 6))
    
def simulate(steadyStateTime, 
    nSims, 
    numThreads, 
    parameterUpdateValues, 
    updatePropensity, 
    populationInit, 
    speciesIndex, 
    baseSeed,
    showPlot = False,
    shouldSave = False,
    savePath = None,
    logQueue = None):
    #parameters for calculation
    timePoints = np.linspace(0, steadyStateTime, steadyStateTime*2)
    simPerThread = int(nSims/numThreads)

    # logQueue.put(f"Starting GillespieSSA with {nSims} simulations using {numThreads} threads.")
    #Running Gillespie simulation
    samples = gillespieSSA(
        updatePropensity,
        parameterUpdateValues,
        populationInit,
        timePoints,
        baseSeed = baseSeed,
        simPerThread= simPerThread,
        numThreads= numThreads,
        progressBar= False,
        logQueue = logQueue
    )
    ###PRINT###
    # logQueue.put(f"Finished GillespieSSA with {nSims} simulations using {numThreads} threads.")

    if showPlot:
            ###PRINT###
        # logQueue.put(f"Plotting the final plots of GillespieSSA with {nSims} simulations using {numThreads} threads.")
        plotSim(samples, timePoints, simPerThread, numThreads, speciesIndex, shouldSave=shouldSave, savePath=savePath)
    return samples

def getAverageExpression(samples, speciesIndex, avgOver=100, maxSlope=1e-2, logQueue=None):
    steadyStateVals = {}
    slopes = {}
    relativeSlopes = {}
    steadyStateReached = True
    
    for species in speciesIndex:
        index = speciesIndex[species]
        steadyStateVals[species] = samples[:,-avgOver:, index].mean()

        if '_mRNA' in species:
            expressionOverTime = samples[:, -avgOver:, index].mean(axis=0)
            slope = np.polyfit(np.linspace(0, 1, len(expressionOverTime)), expressionOverTime, 1)[0]
            slopes[species] = slope
            
            relativeSlope = slope / steadyStateVals[species] if steadyStateVals[species] != 0 else float('inf')
            relativeSlopes[species] = relativeSlope

            if abs(relativeSlope) > maxSlope and abs(slope) > 5:
                steadyStateReached = False
                # if logQueue:
                #     logQueue.put(f'WARNING: Relative slope for {species} is {relativeSlope:.4f}, which exceeds the maximum of {maxSlope}')

    if not steadyStateReached:
        if logQueue:
            logQueue.put('ERROR: Steady state not reached. Printing all slopes and relative slopes:')
            for species in slopes:
                logQueue.put(f'{species}:')
                logQueue.put(f'  Slope: {slopes[species]:.4e}')
                logQueue.put(f'  Relative Slope: {relativeSlopes[species]:.4e}')
        return -1, -1
    return steadyStateVals, slopes

def getAverageExpressionTemp(samples, speciesIndex, avgOver=100, maxSlope=1e-2, logQueue=None):
    steadyStateVals = {}
    slopes = {}
    relativeSlopes = {}
    steadyStateReached = True
    steadyStateDistribution = {}
    for species in speciesIndex:
        index = speciesIndex[species]
        steadyStateVals[species] = samples[:,-avgOver:, index].mean()
    
        if '_mRNA' in species:
            expressionOverTime = samples[:, -avgOver:, index].mean(axis=0)
            slope = np.polyfit(np.linspace(0, 1, len(expressionOverTime)), expressionOverTime, 1)[0]
            slopes[species] = slope
            steadyStateDistribution[species] = samples[:, -1, index]
            relativeSlope = slope / steadyStateVals[species] if steadyStateVals[species] != 0 else float('inf')
            relativeSlopes[species] = relativeSlope

            if abs(relativeSlope) > maxSlope and abs(slope) > 5:
                steadyStateReached = False
                # if logQueue:
                #     logQueue.put(f'WARNING: Relative slope for {species} is {relativeSlope:.4f}, which exceeds the maximum of {maxSlope}')

    if not steadyStateReached:
        if logQueue:
            logQueue.put('ERROR: Steady state not reached. Printing all slopes and relative slopes:')
            for species in slopes:
                logQueue.put(f'{species}:')
                logQueue.put(f'  Slope: {slopes[species]:.4e}')
                logQueue.put(f'  Relative Slope: {relativeSlopes[species]:.4e}')
        return -1, -1, -1
    return steadyStateVals, slopes, steadyStateDistribution

def fitLognorm(data, scale = None):
    return lognorm.fit(data, floc=0, fscale = scale)

def fitNegativeBinomial(data):
    def negativeBinomialNLL(params, data):
        mu, alpha = params
        p = 1 / (1 + alpha * mu)
        n = 1 / alpha
        return -np.sum(nbinom.logpmf(data, n, p))
    
    mean = np.mean(data)
    var = np.var(data)
    
    # Initial guess
    alpha_guess = (var - mean) / (mean ** 2) if var > mean else 1e-5
    initial_params = [mean, alpha_guess]
    
    # Bounds to ensure valid parameter values
    bounds = [(1e-5, np.inf), (1e-10, np.inf)]
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            result = minimize(
                negativeBinomialNLL,
                initial_params,
                args=(data,),
                method='L-BFGS-B',
                bounds=bounds
            )
        
        if result.success:
            mu, alpha = result.x
            p = 1 / (1 + alpha * mu)
            n = 1 / alpha
            return n, p
        else:
            raise ValueError("Optimization failed")
    except:
        # If optimization fails, return a very dispersed negative binomial (close to Poisson)
        print("Error")
        return mean * 1e5, 0.99999

def getDistributionParametersNB(dataset, speciesIndex, logQueue=None):
    n = {}  # Number of successes
    p = {}  # Probability of success
    otherMu = {}
    
    for species in speciesIndex:
        index = speciesIndex[species]
        data = dataset[:, :, index].flatten()
        
        if '_mRNA' in species:
            # Filter out zeros and round to nearest integer
            data = np.round(data).astype(int)
            
            # Fit negative binomial distribution
            n_fit, p_fit = fitNegativeBinomial(data)
            n[species] = n_fit
            p[species] = p_fit
            
            if logQueue:
                logQueue.put(f"Fitted NB for {species}: n={n_fit:.4f}, p={p_fit:.4f}")
        else:
            otherMu[species] = np.mean(data)
            
            if logQueue:
                logQueue.put(f"Mean for {species}: {otherMu[species]:.4f}")
    
    return n, p, otherMu

def getDistributionParameters(dataset, speciesIndex, logQueue = None):
    mu = {}
    sigma = {}
    otherMu = {}
    for species in speciesIndex:
        if '_mRNA' in species:
            index = speciesIndex[species]
            data = dataset[:, :, index].flatten()
            data = data[data>0]
            mean = np.mean(np.log(data))
            shape, _, scale = fitLognorm(data, scale = None)
            mu[species] = scale
            sigma[species] = shape
            # print(f"mu: {mu}, sigma: {sigma}")
        else:
            index = speciesIndex[species]
            otherMu[species] = dataset[:, :, index].flatten().mean()
            # print(f"otherMu: {otherMu}")
    return mu, sigma, otherMu

def divideRNAWithNoise(geneExpression, rng, allowNegative=False):
    # Separate the keys into those containing mRNA and those that should be copied directly
    mRNAValues = {key: value for key, value in geneExpression.items() if key.endswith('_mRNA')}
    geneStates = {key: value for key, value in geneExpression.items() if not key.endswith('_mRNA')}

    # Initialize two dictionaries for cell 1 and cell 2
    resultDictCell1 = geneStates.copy()
    resultDictCell2 = geneStates.copy()

    # Calculate statistics for noise factor
    mRNACounts = np.array(list(mRNAValues.values()))
    mean = np.mean(mRNACounts)
    stdDev = np.std(mRNACounts)

    def calculateNoiseFactor(zScore):
        return min(abs(zScore) * 0.1, 0.3)

    for key, value in mRNAValues.items():
        if value == 0:
            resultDictCell1[key] = 0
            resultDictCell2[key] = 0
            continue

        zScore = (value - mean) / stdDev if stdDev != 0 else 0
        noiseFactor = calculateNoiseFactor(zScore)
        
        halfValue = value / 2
        noise = rng.uniform(-noiseFactor, noiseFactor) * halfValue
        
        split1 = int(round(halfValue + noise))
        split1 = max(0, split1)
        split2 = value - split1
        split2 = max(0, split2)

        resultDictCell1[key] = split1
        resultDictCell2[key] = split2

    return resultDictCell1, resultDictCell2

def divideRnaEqual(geneExpression, rng, probPairing=0.999):
    # Separate the keys into those containing mRNA and those that should be copied directly
    mRNAValues = {key: value for key, value in geneExpression.items() if key.endswith('_mRNA')}
    geneStates = {key: value for key, value in geneExpression.items() if not key.endswith('_mRNA')}

    # Initialize two dictionaries for cell 1 and cell 2
    resultDictCell1 = geneStates.copy()
    resultDictCell2 = geneStates.copy()

    # Calculate statistics for noise factor
    mRNACounts = np.array(list(mRNAValues.values()))
    
    for key, parentCount in mRNAValues.items():
        

        if parentCount == 0:
            resultDictCell1[key] = 0
            resultDictCell2[key] = 0
            continue
        # Calculate paired counts and check if they are odd
        pairedCounts = int(np.round(probPairing * parentCount))
        
        # Divide paired counts evenly and randomly assign extra if odd
        if pairedCounts % 2 == 1:
            if rng.random() > 0.5:
                sisterCell1 = pairedCounts // 2 + 1
                sisterCell2 = pairedCounts // 2
            else:
                sisterCell1 = pairedCounts // 2
                sisterCell2 = pairedCounts // 2 + 1
        else:
            sisterCell1 = sisterCell2 = pairedCounts // 2
        # Calculate remaining counts and distribute with binomial distribution
        remainingCounts = parentCount - pairedCounts
        if remainingCounts > 0:
            sisterCell1 += rng.binomial(remainingCounts, 0.5)
        
        # Ensure that sisterCell2 is the complement to sisterCell1 for the total parentCount
        sisterCell2 = parentCount - sisterCell1
        
        # Store the counts in the result dictionaries
        resultDictCell1[key] = sisterCell1
        resultDictCell2[key] = sisterCell2
    
    return resultDictCell1, resultDictCell2

#New: instead of dividing, what if I make it a copy
def copyParentIntoDaughter(geneExpression, rng, noiseLevel = 0.01):
    mRNAValues = {key: value for key, value in geneExpression.items() if key.endswith('_mRNA')}
    geneStates = {key: value for key, value in geneExpression.items() if not key.endswith('_mRNA')}

    # Initialize two dictionaries for cell 1 and cell 2
    resultDictCell1 = geneStates.copy()
    resultDictCell2 = geneStates.copy()

    # Calculate statistics for noise factor
    mRNACounts = np.array(list(mRNAValues.values()))
    
    for key, parentCount in mRNAValues.items():
        

        if parentCount == 0:
            resultDictCell1[key] = 0
            resultDictCell2[key] = 0
            continue
        noise1 = rng.normal(0, noiseLevel * parentCount)
        noise2 = rng.normal(0, noiseLevel * parentCount)
        sisterCell1 = max(0, round(parentCount + noise1))
        sisterCell2 = max(0, round(parentCount + noise2))
        resultDictCell1[key] = sisterCell1
        resultDictCell2[key] = sisterCell2

    return resultDictCell1, resultDictCell2

def divideRNA(geneExpression, rng):
    # Separate the keys into those containing mRNA and those that should be copied directly
    mRNAValues = {key: value for key, value in geneExpression.items() if key.endswith('_mRNA')}
    # parent cell gene states are being maintained
    geneStates = {key: value for key, value in geneExpression.items() if not key.endswith('_mRNA')}

    # Initialize two dictionaries for cell 1 and cell 2
    resultDictCell1 = geneStates.copy()
    resultDictCell2 = geneStates.copy()

    for key, value in mRNAValues.items():
        binomialSamples = rng.binomial(1, 0.5, size=value)
        cell1Count = np.sum(binomialSamples == 0)
        cell2Count = np.sum(binomialSamples == 1)

        resultDictCell1[key] = cell1Count
        resultDictCell2[key] = cell2Count
    return resultDictCell1, resultDictCell2


class Cell:
    nextId = 0
    speciesIndex = {}
    updatePropensity = None
    parameterUpdateValues = None
    propensityParameters = None
    propensities = None

    def __init__(self, expression, clock, simPerThread, numThreads, samples = None, splitFrom = None, speciesIndex = None):
        self.geneExpression = expression
        self.splitFrom = splitFrom
        self.clock = clock
        self.samples = samples
        self.numThreads = numThreads
        self.simPerThread = simPerThread
        self.id = Cell.nextId
        if(speciesIndex):
            self.speciesIndex = speciesIndex
        Cell.nextId +=1

    def __str__(self):
        return 'Cell {}'.format(self.id)
    
    def runSimulation(self, time, showPlot = False, baseSeed = None, speciesIndex = None, updatePropensity = None, parameterUpdateValues = None, logQueue = None, maintain = False):
        timePoints = np.linspace(self.clock, self.clock + time, time *2)
        #Setting up the initial gene/mRNA expression
        # logQueue.put(f"In runSimulation for cell {self.id}")
        if speciesIndex is not None and not np.any(speciesIndex == None):
            speciesIndexToProvide = speciesIndex
        else:
            speciesIndexToProvide = self.speciesIndex

        if parameterUpdateValues is not None and not np.any(parameterUpdateValues == None):
            parameterUpdateValuesToProvide = parameterUpdateValues
        else:
            parameterUpdateValuesToProvide = self.parameterUpdateValues

        if updatePropensity is not None and not np.any(updatePropensity == None):
            updatePropensityToProvide = updatePropensity
        else:
            updatePropensityToProvide = self.updatePropensity

        populationExpression = Cell.setupStartingExpression(self.geneExpression, speciesIndexToProvide , maintain = maintain, logQueue=logQueue)
        # logQueue.put("line 5")
        # logQueue.put(f"Starting gillespieSSA for cell {self.id}")
        newSamples = gillespieSSA(
            updatePropensity = updatePropensityToProvide,
            parameterUpdateValues = parameterUpdateValuesToProvide,
            populationInit = populationExpression,
            timePoints= timePoints,
            simPerThread = self.simPerThread,
            numThreads = self.numThreads,
            progressBar = False,
            baseSeed=baseSeed, 
            logQueue = logQueue
        )

        if self.samples is None:
            self.samples = newSamples
        else:
            self.samples = np.concatenate([self.samples, newSamples], axis = 1)
        
        self.geneExpression = self.getPointExpression(time = -1, speciesIndex=speciesIndexToProvide)
        self.clock += time
        # print(f"In runSimulation for {self.id}: geneExpression: {self.geneExpression}, self.samples = {self.samples}")
        if showPlot:
            plotSim(
                self.samples,
                np.linspace(0, self.clock, self.clock*2),
                self.simPerThread,
                self.numThreads,
                Cell.speciesIndex,
                trendline = False,
            )
        
        # logQueue.put(f"Finished runSimulation for cell {self.id}")
        return self


    def cellSplit(self, rng, numThreads = None):
        #at this point, all of these cell-information is retained well.
        # print(self.clock, self.geneExpression, self.parameterUpdateValues, self.propensityParameters)
        # geneExpression1, geneExpression2 = divideRnaEqual(self.geneExpression, rng)
        #Changed code #NEW - instead of making it half, copied it
        geneExpression1, geneExpression2 = copyParentIntoDaughter(self.geneExpression, rng)

        if (numThreads!=None):
            numThreadsToPass = numThreads
        else:
            numThreadsToPass = self.numThreads
        newCell1 = Cell(
            geneExpression1,
            clock = self.clock,
            simPerThread= self.simPerThread,
            numThreads= numThreadsToPass,
            splitFrom= self.id,
            speciesIndex = Cell.speciesIndex
        )
        #New cell 1 has all these properties
        # print(f"New cell 1: geneExpression1: {geneExpression1}\n newCell1.propensityParameters: {newCell1.propensityParameters}")
        newCell2 = Cell(
            geneExpression2,
            clock = self.clock,
            simPerThread= self.simPerThread,
            numThreads= numThreadsToPass,
            splitFrom= self.id,
            speciesIndex = Cell.speciesIndex
        )
        #instantiate samples with 0s for all genes
        newCell1.samples = np.zeros((len(self.samples), len(self.samples[0]), len(self.geneExpression)))
        newCell2.samples = np.zeros((len(self.samples), len(self.samples[0]), len(self.geneExpression)))
        return newCell1, newCell2
    

    def getPointExpression(self, sample=None, time = -1, speciesIndex = "ABCD"):
        sample = self.samples if sample is None else sample
        expr = {}
        pos = int(time * 2) if time!= -1 else -1
        if sample.shape[0] != 1:
            raise(ValueError('Samples must be 1D array'))
        if(speciesIndex!= "ABCD"):
            speciesIndexCell = speciesIndex
        else:
            speciesIndexCell = Cell.speciesIndex
        for species in speciesIndexCell:
            index = speciesIndexCell[species]
            expr[species] = int(sample[0,pos, index])
        return expr
        
    @classmethod
    def resetId(cls):
        cls.nextId = 0
        
    @classmethod
    def setSimulationParameters(cls, speciesIndex, propensities, propensityParameters, parameterUpdateValues=None, updatePropensity = None):
        cls.propensities = propensities
        cls.propensityParameters = propensityParameters
        cls.parameterUpdateValues = parameterUpdateValues
        cls.updatePropensity = updatePropensity
        cls.speciesIndex = speciesIndex

    @classmethod
    def setupStartingExpression(cls, geneExpression, speciesIndex, maintain=False, logQueue="abcd"):
        # if logQueue != "abcd":
        #     logQueue.put("In setupStartingExpression")
        
        # Initialize populationExpression with zeros
        populationExpression = [0] * len(speciesIndex)
        
        if maintain:
            # If cells are dividing, they will inherit the gene state from parent
            for gene in geneExpression:
                expr = geneExpression[gene]
                populationExpression[speciesIndex[gene]] = expr
            populationExpression = np.array(populationExpression, dtype=np.int64)
        else: 
            for gene in geneExpression:
                # if logQueue != "abcd":
                #     logQueue.put("line - 1")
                # print(gene)
                expr = geneExpression[gene]
                # if logQueue != "abcd":
                #     logQueue.put("line - 2")

                # Debug prints to trace the issue
                # print(f"Processing gene: {gene}")
                # print(f"speciesIndex: {speciesIndex}")
                # print(f"Index for gene '{gene}': {speciesIndex[gene]}")
                # print(f"Current populationExpression: {populationExpression}")
                
                # Setting genes to on or off based on steady state probabilities
                if '_A' == gene[-2:]:
                    populationExpression[speciesIndex[gene]] = 0 if expr <= 0.5 else 1
                elif '_I' == gene[-2:]:
                    populationExpression[speciesIndex[gene]] = 0 if expr <= 0.5 else 1
                else:
                    populationExpression[speciesIndex[gene]] = expr

                # if logQueue != "abcd":
                #     logQueue.put("line - 3")
            
            populationExpression = np.array(populationExpression, dtype=np.int64)
        
        # if logQueue != "abcd":
        #     logQueue.put("Finished setupStartingExpression")
        
        return populationExpression
    
    @classmethod
    def generateLognormal(cls, mu, sigma, size, generator):
        expressions = np.empty((size, len(mu)))
        for i, species in enumerate(mu.keys()):
            expressions[:, i] = np.round(generator.lognormal(mean=np.log(mu[species]), sigma=sigma[species], size=size))
        expressionDf = pd.DataFrame(expressions, columns=mu.keys())
        expressionDict = expressionDf.to_dict(orient='records')
        # print(len(expressionDict))
        return expressionDict
    
    @classmethod
    def generateNegativeBinomial(cls, n, p, size, generator):
        expressions = np.empty((size, len(n)))
        for i, species in enumerate(n.keys()):
            nValue = n[species]
            pValue = p[species]
            
            # Generate from negative binomial distribution
            expressions[:, i] = generator.negative_binomial(n=nValue, p=pValue, size=size)
        
        expressionDf = pd.DataFrame(expressions, columns=n.keys())
        expressionDict = expressionDf.to_dict(orient='records')
        return expressionDict

    @classmethod
    def createCells(cls, nCells, mu, sigma, geneState, baseSeed, logQueue=None):
        try:
            cells = []
            generator = np.random.default_rng(baseSeed)
            
            # if logQueue:
            #     logQueue.put("Entered createCells")
    # Convert the DataFrame to a list of dictionaries
            try:
                expression = Cell.generateLognormal(mu, sigma, nCells, generator)
            except KeyError as e:
                error_message = f"Error: Missing distribution parameters for {str(e)}"
                if logQueue:
                    logQueue.put(error_message)
                raise ValueError(error_message) from None

            # if logQueue:
            #     logQueue.put("Generated expression for all cells")

            for i in range(nCells):
                # currentCellExpression = {idx: value for idx, value in enumerate(expression[i].tolist())}
                # print( geneState)
                try:
                    expressionDict = {**expression[i], **geneState }
                    # print(expressionDict)
                    # currentCellExpression = Cell.setupStartingExpression(geneExpression = expressionDict, speciesIndex=Cell.speciesIndex, logQueue=logQueue)
                    # logQueue.put("returned from setupStartingExpression")
                    # print(currentCellExpression)
                    cells.append(cls(expression=expressionDict, clock=0, simPerThread=1, numThreads=1, speciesIndex=Cell.speciesIndex))

                    # if logQueue:
                    #     logQueue.put("Done creating cells")
                except Exception as e:
                    logQueue.put(f"Unhandled exception in createCells: {str(e)}")
                    raise
            
            return cells
            

        except Exception as e:
            if logQueue:
                logQueue.put(f"Unhandled exception in createCells: {str(e)}")
            raise
    
    @classmethod
    def createBulkCells(cls, nCells, time, expression,
                         numThreads, showPlot = False,
                           baseSeed = None, logQueue = None):
        timePoints = np.linspace(0, time, time*2)
        #setup new gene expression array
        populationExpression = Cell.setupStartingExpression(expression, Cell.speciesIndex)
        simPerThread = int(nCells/numThreads)
        # logQueue.put(f"Created {nCells} cells and now running gillespieSSA on them using {numThreads} threads")
        samples = gillespieSSA(
            Cell.updatePropensity,
            Cell.parameterUpdateValues,
            populationExpression,
            timePoints,
            simPerThread = simPerThread,
            numThreads = numThreads,
            progressBar = False,
            baseSeed=baseSeed,
            logQueue= logQueue
        )
        # logQueue.put('Finished simulating {} cells'.format(simPerThread*numThreads))

        print('Finished creating {} cells'.format(simPerThread*numThreads))
        if showPlot:
            plotSim(samples, timePoints, simPerThread, numThreads, Cell.speciesIndex)
        
        cellFinal = []
        for sample in samples:
            sample = np.array(sample, ndmin=3)
            expression = Cell.getPointExpression(cls, sample, time = -1)
            cellFinal.append(Cell(expression, clock = time, simPerThread=1, numThreads=1, samples=sample))
        
        # logQueue.put('Returning final {} cells'.format(simPerThread*numThreads))
        return cellFinal
        
    @classmethod
    def sortByExpression(cls, cellFinal, gene, time = -1):
        if time == -1:
            return sorted(cellFinal, key = lambda x: x.geneExpression[gene], reverse=True)
        else:
            return sorted(cellFinal, key=lambda x:x.samples[0, time*2, Cell.speciesIndex[gene]], reverse=True)
