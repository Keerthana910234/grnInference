#This is the script that will handle the simulation per pair of parameter amd reaction set
import numpy as np
import tqdm
import pandas as pd
import dill
import os
import datetime
import zstandard as zstd
import numpy as np
import numba
import bokeh.plotting
import multiprocessing
from concurrent.futures import ProcessPoolExecutor 
from multiprocessing.managers import SyncManager
from simulationFunctions_perturbed import Cell, setupGillespieParams, simulate, getAverageExpression, getDistributionParameters
# from memory_profiler import profile

def simulateCell(cellData):
    cell, time, showPlot, baseSeed, speciesIndex, updatePropensity, parameterUpdateValues,  logQueue, maintain, speciesToPerturb, timeOfPerturbation, amountToPerturb = cellData
    # logQueue.put("Entered simulateCell")
    # print(f"Time in simulateCell(should be 500): {cell.clock}")
    cell = cell.runSimulation(time=time, showPlot=showPlot, baseSeed=baseSeed, speciesIndex=speciesIndex, updatePropensity = updatePropensity, parameterUpdateValues= parameterUpdateValues, logQueue = logQueue, maintain = maintain, speciesToPerturb = speciesToPerturb, timeOfPerturbation = timeOfPerturbation, amountToPerturb=amountToPerturb)
    # logQueue.put("Exited simulateCell")
    # print(f"Time in simulateCell(should be 2000): {cell.clock}")
    return cell

# @profile
def createSimulation(initialStates, reactions, propensityParameters, steadyStateTime,numThreads, seed, initialTime, postSplitTime, nCells, outputPath, nSims, parameterRow, graphName, logQueue, speciesToPerturb = None, timeOfPerturbation = None, amountToPerturb = None):
    logQueue.put(f"Simulation started for graph: {graphName} with parameters from {parameterRow}")
    #Setting up gillespie parameters
    populationInit, parameterUpdateValues, updateFunc, speciesIndex = \
    setupGillespieParams(
        initialStates, reactions, propensityParameters, logQueue = logQueue
    )
    exec(updateFunc, globals())

    print("Running simulation to steady state")
    # To create a cache before multiprocessing
    # logQueue.put("line 1 started")
    
    samplesNone = simulate(
    steadyStateTime = 125, 
    nSims = 1, 
    numThreads = 1, 
    parameterUpdateValues = parameterUpdateValues, 
    updatePropensity = updatePropensity, 
    populationInit = populationInit, 
    speciesIndex = speciesIndex, 
    baseSeed = 101,
    showPlot = False,
    shouldSave = False,
    savePath = outputPath, 
    logQueue = logQueue
    )
    # logQueue.put("line 1 ended")
    now = datetime.datetime.now()
    formattedDatetime = now.strftime("%Y%m%d_%H%M%S")
    #Create folders if it does not exist to store the steady state data
    os.makedirs(f'{outputPath}/steadyStateData/', exist_ok=True)

    samples = simulate(
        steadyStateTime, 
        nSims, 
        numThreads, 
        parameterUpdateValues, 
        updatePropensity, 
        populationInit, 
        speciesIndex, 
        showPlot = True,
        shouldSave = True,
        baseSeed = seed[0],
        savePath= f"{outputPath}/steadyStateData/parameterRow{parameterRow}_Graph{graphName}.png",
        logQueue = logQueue
    )
    runName = f"parameterRow{parameterRow}_Graph{graphName}"
    samplesSave = np.array(samples)
    np.savez_compressed(
        f"{outputPath}/steadyStateData/{runName}_samples.npz",
        data=samplesSave,
    )
    # logQueue.put("line 2 ended")

    # logQueue.put("Testing for steady state")
    steadyState, slopes = getAverageExpression(samples, speciesIndex, 
                                               avgOver = 500, maxSlope=5,        
                                               logQueue = logQueue)
    if steadyState == -1 and slopes == -1:
        logQueue.put(f"ERROR in reaching steady state for parameter {parameterRow} for Graph {graphName}")
        return
    # logQueue.put(f"Slopes obtained were: {slopes}")
    # del samples
    logQueue.put("Steady State has been reached if no warnings are generated")
    print("Steady State has been reached if no warnings are generated")
    datasetForFitting =  samples[:,-500:, :]
    mu, sigma, geneState = getDistributionParameters(datasetForFitting, speciesIndex, logQueue = logQueue)
    del samples
    #Sets the properties for all of cell class - so important that this is set in each process
    Cell.setSimulationParameters(speciesIndex, reactions, propensityParameters, parameterUpdateValues, updatePropensity)
    #reset cell Id
    Cell.resetId()

    #Creating cells
    cells = []

    ### TODO : modify the function to have heterogenous cells

    # cells = Cell.createBulkCells(nCells, initialTime, steadyState, numThreads = numThreads, showPlot = False, baseSeed = seed[1], logQueue = logQueue)
    cells = Cell.createCells(nCells, mu, sigma, geneState, seed[1], logQueue=logQueue)

    numProcesses = numThreads # Number of processes
    # logQueue.put(f"Preparing {len(newCells)} tuples for each cell containing necessary parameters")
    # Prepare a tuple for each cell containing necessary parameters
    tasksInit = [(cell, initialTime, False, seed[1], speciesIndex, updatePropensity, parameterUpdateValues,  logQueue, False, None, np.inf, 1) for cell in cells]
    # Cell.setSimulationParameters(speciesIndex, reactions, propensityParameters, parameterUpdateValues, updatePropensity)

    # Create a multiprocessing pool
    # print("Simulating cells after initiation")
    with ProcessPoolExecutor(max_workers=numProcesses) as pool:
        # Map the function over the cells
        #Now collecting newCells after running to see if stores anything
            cells = list(pool.map(simulateCell, tasksInit))
    logQueue.put("Dividing cells")
    newCells = []
    rngSplit = np.random.default_rng(seed[1])
    for i, cell in enumerate(cells):
        splitCells = cell.cellSplit(rngSplit, numThreads = 1)
        if isinstance(splitCells, (tuple, list)):
            newCells.extend(splitCells)
        else:
            newCells.append(splitCells)
    
    if 'tasksInit' in locals():
        del tasksInit
    # newCell1 = newCells[2]
    # print(f"New cell 1a: geneExpression1: {newCell1.geneExpression}\n newCell1.propensityParameters: {newCell1.propensityParameters}")
    # cells.extend(newCells)

    # for i, cell in enumerate(tqdm.tqdm(newCells, "Continue simulation")):
    #     cell.runSimulation(time = postSplitTime, showPlot = False, baseSeed = seed[2])

    ###BETA CODE###
    # logQueue.put(f"Beginning the Beta code to start multi-process for simulating cells with {numThreads} threads")
    # numProcesses = numThreads # Number of processes
    # logQueue.put(f"Preparing {len(newCells)} tuples for each cell containing necessary parameters")
    # Prepare a tuple for each cell containing necessary parameters
    tasks = [(cell, postSplitTime, False, seed[2], speciesIndex, updatePropensity, parameterUpdateValues,  logQueue, True, speciesToPerturb, timeOfPerturbation if idx % 2 == 1 else np.inf, amountToPerturb) for idx, cell in enumerate(newCells)]
    # Cell.setSimulationParameters(speciesIndex, reactions, propensityParameters, parameterUpdateValues, updatePropensity)

    # Create a multiprocessing pool
    # print("Continuing simulation after split")
    with ProcessPoolExecutor(max_workers=numProcesses) as pool:
        # Map the function over the cells
        #Now collecting newCells after running to see if stores anything
            newCells = list(pool.map(simulateCell, tasks))

    if 'tasks' in locals():
        del tasks
    # print("Last checkpoint before saving")
    # newCell1 = newCells[2]
    # print(f"New cell 1: geneExpression1: {newCell1.geneExpression}\n newCell1.propensityParameters: {newCell1.propensityParameters}")
    # Saving the simulation
    now = datetime.datetime.now()
    runName = f"parameterRow{parameterRow}_Graph{graphName}_cells{len(cells)}_" + now.strftime("%Y-%m-%d-%H-%M-%S")

    # Initialize ZSTD compression context with parameters
    outputFilePath = f"{outputPath}/{runName}perturbed_{speciesToPerturb}_factorPerturb_{amountToPerturb}.pkl.zst"
    logQueue.put("Saving now")
    with open(outputFilePath, "wb") as outputFile:
        with zstd.ZstdCompressor(
                level=9, threads=numThreads,
                write_checksum=True).stream_writer(outputFile) as compressor:
            # Stream the serialization directly to the compressor
            dill.dump([
                    cells,
                    newCells,
                    initialTime,
                    postSplitTime,
                    speciesIndex,
                    {"initialStates": initialStates, "reactions":reactions, "propensityParameters": propensityParameters},
                    {
                        "steadyState": steadyState,
                        "nSimulations": nSims,
                        "steadyStateTime": steadyStateTime,
                        "steadyStateSlope": slopes,
                        "amountToPerturb": amountToPerturb
                    },
                ], compressor)
    print("Saved as " + outputFilePath)
    print(os.path.getsize(outputFilePath) / 1000000, "MB")
    #log_detailed_memory(logQueue, "After saving")
    # # copy that file to long term storage
    logQueue.put(
        f"Simulation finished for graph: {graphName} with parameters from {parameterRow}"
    )
    
