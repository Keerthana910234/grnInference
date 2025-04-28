from simulationPerSet import createSimulation
import os
import pandas as pd
from numpy.random import default_rng
import multiprocessing 
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import List, Tuple
from logging import handlers
from time import sleep
from  multiprocessing.queues import Queue
import logging
from logging.handlers import QueueHandler, RotatingFileHandler
import multiprocessing
from multiprocessing.managers import SyncManager
import datetime
import traceback
import sys
import gc
import psutil

import re
from typing import Set

def log_detailed_memory(logQueue, stage):
    """Log detailed memory information"""
    process = psutil.Process()
    
    # Main process memory
    main_memory = process.memory_info().rss / (1024 * 1024 * 1024)
    
    # Child process memory
    child_memory = 0
    for child in process.children(recursive=True):
        try:
            child_memory += child.memory_info().rss / (1024 * 1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Log the results
    logQueue.put(f"\n=== Memory Usage at {stage} ===")
    logQueue.put(f"Main process memory: {main_memory:.2f} GB")
    logQueue.put(f"Child processes memory: {child_memory:.2f} GB")
    logQueue.put(f"Total memory: {(main_memory + child_memory):.2f} GB")
    logQueue.put("================================\n")

def startManager():
    m = SyncManager()
    m.start()
    return m

def logException(exc, logQueue):
    # Get the full traceback as a string
    traceback_str = ''.join(traceback.format_exception(None, exc, exc.__traceback__))
    
    # Log the exception with the full traceback
    logQueue.put(f"A task generated an exception: {exc}\n{traceback_str}")

def setupLogging(queue):
    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Set the minimum level of logs for the logger

    # Setup QueueHandler to send logs to the queue
    qh = QueueHandler(queue)
    logger.addHandler(qh)

def logListenerProcess(queue):
    # Setup the logging handler and formatter only once outside the loop
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")
    logPath = "/home/mzo5929/Keerthana/GRNsimulation/highthroughputData/linearNetworkSim/logs/"
    os.makedirs(logPath, exist_ok=True)
    handler = RotatingFileHandler(f'{logPath}/logfile_{formatted_datetime}.log', maxBytes=1048576, backupCount=40)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger("LogListener")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    while True:
        message = queue.get()
        if message is None:  # Sentinel value to exit
            break
        # Log the message received as an INFO level log
        logger.info(message)

    # Clean up handler resources
    logger.removeHandler(handler)
    handler.close()

def getParameterList(directory: str) -> Set[int]:
    """Extract all unique parameter numbers from filenames."""
    pattern = re.compile(r'parameterRow(\d+(?:\.\d+)?)_Graph\d+_cells10000\.pkl\.zst$')
    parameterSet = set()
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            parameter_num = int(float(match.group(1)))
            parameterSet.add(parameter_num)
    
    return parameterSet

def findMissingParameters(existingParams: Set[int], start: int = 0, end: int = 10500) -> Set[int]:
    """Find missing parameter numbers in the specified range."""
    fullRange = set(range(start, end))
    return fullRange - existingParams

def readInputData(reactionsFolder: str, parameterSheet: str, initialStatesPath: str, logQueue: Queue, baseSeed: int = 101010, networkNumber: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame, List[int], pd.DataFrame]]:
    """
    Reads reaction data, parameters, and initial states, and generates random seeds for simulations.
    
    Args:
        reactionsFolder (str): Path to the folder containing reaction files.
        parameterSheet (str): Path to the CSV file containing parameters.
        initialStatesPath (str): Path to the CSV file containing initial states.
        baseSeed (int): Seed for the random number generator.

    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame, List[int], pd.DataFrame]]: List of tuples containing 
            reaction data, parameters, seeds, and initial states for simulations.
    """
    rng = default_rng(baseSeed)
    logQueue.put("Creating Input Pairs")
    reactionFiles = [f"{reactionsFolder}/reactions.csv"]
    if reactionFiles:
        reactionFile = reactionFiles  # Safely access the first file
    else:
        logQueue.put("No CSV files found in the directory")
        return []  # Return an empty list or handle this case appropriately
    reactionFiles.sort()
    parameterData = pd.read_csv(parameterSheet, index_col=0)
    parameterData['originalIndex'] = parameterData.index
    initialStates = pd.read_csv(initialStatesPath, index_col=False)  
    # existingParams = getParameterList("/home/mzo5929/Keerthana/GRNsimulation/highthroughputData/linearNetworkSim/3/")
    existingParams = set()
    print(f"\nFound {len(existingParams)} completed parameter files")
    missingParams = findMissingParameters(existingParams)
    pairs = []
    numSeedsPerCombination = 3
    parameterData = parameterData.loc[list(missingParams)].reset_index(drop = True)
    seedsAll = rng.integers(low=0, high=2**32, size=numSeedsPerCombination*len(reactionFiles)*parameterData.shape[0]).tolist()
    index = 0
    for i, reactionFile in enumerate(reactionFiles):
        reactionDataFrame = pd.read_csv(reactionFile)
        for paramRow, parameter in parameterData.iterrows():
            
            # Use the original index preserved earlier
            originalParamRow = int(parameter['originalIndex'])  # Transpose to make it a single row DataFrame
            singleRowDf = parameter.drop(labels=['originalIndex']).to_frame().T
            meltedParameterDf = singleRowDf.melt(var_name='parameter', value_name='value')
            # formula = "(0.5 * {d} * ({rProd}/{rDeg}))"
            # newRow = pd.DataFrame({
            # 'parameter': ['k'], 
            # 'value': [formula]
            # })
            # meltedParameterDf = pd.concat([newRow,meltedParameterDf], ignore_index=True)
            startIndex = (i * len(parameterData) + index) * numSeedsPerCombination
            # print(paramRow)
            seeds = seedsAll[startIndex:startIndex + numSeedsPerCombination]
            pairs.append((reactionDataFrame, meltedParameterDf, seeds, initialStates,originalParamRow, str(networkNumber)))
            index = index + 1
    # for index, parameter in parameterData.iterrows():

    #         singleRowDf = parameter.to_frame().T  # Transpose to make it a single row DataFrame
    #         meltedParameterDf = singleRowDf.melt(var_name='parameter', value_name='value')
    #         formula = "(0.95 * {d} * ({rProd}/{rDeg}))"
    #         newRow = pd.DataFrame({
    #         'parameter': ['k'], 
    #         'value': [formula]
    #         })
    #         meltedParameterDf = pd.concat([newRow,meltedParameterDf], ignore_index=True)
    #         for reactionFile in reactionFiles:
    #             reactionDataFrame = pd.read_csv(reactionFile)
    #             seeds = rng.integers(low=0, high=2**32, size=3).tolist()
    #             pairs.append((reactionDataFrame, meltedParameterDf, seeds, initialStates,index, os.path.splitext(os.path.basename(reactionFile))[0]))
    
    del seedsAll
    del parameterData
    gc.collect()
    pairs.sort(key=lambda x: x[4])
    logQueue.put("Created Input Pairs")
    return pairs

def runSimulation(pair: Tuple[pd.DataFrame, pd.Series, List[int], pd.DataFrame, Queue]) -> None:
    """
    Runs a simulation based on provided reaction data, parameters, and initial states.

    Args:
        pair (Tuple[pd.DataFrame, pd.Series, List[int], pd.DataFrame]): Tuple containing reaction data,
            parameters, seeds, and initial states.
    """
    reaction, parameter, seeds, initialStates, parameterRow, graphName, logQueue = pair
    createSimulation(
        initialStates,
        reactions=reaction,
        propensityParameters=parameter,
        steadyStateTime=2000,
        numThreads=6,
        seed=seeds,
        initialTime=100,
        postSplitTime=2000,
        nCells=10000,
        outputPath=f"/home/mzo5929/Keerthana/GRNsimulation/highthroughputData/linearNetworkSim/{graphName}/",
        nSims=2000,
        parameterRow = parameterRow,
        graphName = graphName, 
        logQueue = logQueue
    )

# def main(reactionsFolder: str, parameterSheet: str, initialStatesPath: str, masterSeed: int, startIndex: int, networkNumber: int):
#     """
#     Main function to process files and run simulations.

#     Args:
#         reactionsFolder (str): Path to reactions folder.
#         parameterSheet (str): Path to parameter sheet CSV.
#         initialStatesPath (str): Path to initial states CSV.
#         masterSeed (int): Master seed for RNG.
#     """
    
#     manager = startManager()
#     logQueue = manager.Queue()
#     listener = multiprocessing.Process(target=logListenerProcess, args=(logQueue,))
#     listener.start()
#     import tracemalloc

#     tracemalloc.start()

#     pairs = readInputData(reactionsFolder, parameterSheet, initialStatesPath, 
#                           baseSeed=masterSeed, logQueue = logQueue, networkNumber=networkNumber)
#     test_pair = pairs[0:3]
#     # print(test_pair[0]) 
#     try:
#         with ProcessPoolExecutor(max_workers=1) as pool:
#             futures = [pool.submit(runSimulation, item + (logQueue,)) for item in test_pair]
#             for future in as_completed(futures):
#                 try:
#                     future.result()  # This will re-raise any exception caught during the task execution
#                 except Exception as exc:
#                     print(f"A task generated an exception: {exc}")
#                     logQueue.put("ERROR TRACE")
#                     logException(exc, logQueue)

#         logQueue.put(None)
#         listener.join()
#         snapshot = tracemalloc.take_snapshot()
#         top_stats = snapshot.statistics('traceback')

#         print("[ Top 10 Memory Consumers (with traceback) ]")
#         for stat in top_stats[:10]:
#             print(stat)
#             for line in stat.traceback.format():
#                 print(line)

#     finally:
#         logQueue.put(None)
#         listener.join()

# def main(reactionsFolder: str, parameterSheet: str, initialStatesPath: str, masterSeed: int, startIndex: int, networkNumber: int):
#     manager = startManager()
#     logQueue = manager.Queue()
#     listener = multiprocessing.Process(target=logListenerProcess, args=(logQueue,))
#     listener.start()
    
#     log_detailed_memory(logQueue, "Start of main")
    
#     pairs = readInputData(reactionsFolder, parameterSheet, initialStatesPath,
#                          baseSeed=masterSeed, logQueue=logQueue, networkNumber=networkNumber)
#     test_pair = pairs[startIndex:]
    
#     log_detailed_memory(logQueue, "After reading input data")
    
#     try:
#         with ProcessPoolExecutor(max_workers=3) as pool:
#             futures = [pool.submit(runSimulation, item + (logQueue,)) for item in test_pair]
            
#             for i, future in enumerate(as_completed(futures)):
#                 log_detailed_memory(logQueue, f"Before processing parameter {i+1}")
#                 try:
#                     result = future.result()
#                     log_detailed_memory(logQueue, f"After processing parameter {i+1}")
#                 except Exception as exc:
#                     print(f"Error in parameter {i+1}: {exc}")
#                     logException(exc, logQueue)
#                 finally:
#                     del future
#                     gc.collect()
#                     if hasattr(os, 'malloc_trim'):
#                         os.malloc_trim(0)
#                     log_detailed_memory(logQueue, f"After cleanup parameter {i+1}")
        
#         log_detailed_memory(logQueue, "After all parameters")
        
#     finally:
#         logQueue.put(None)
#         listener.join()

#Low memory variant
def main(reactionsFolder: str, parameterSheet: str, initialStatesPath: str, masterSeed: int, startIndex: int, networkNumber: int):
    manager = startManager()
    logQueue = manager.Queue()
    listener = multiprocessing.Process(target=logListenerProcess, args=(logQueue,))
    listener.start()

    pairs = readInputData(reactionsFolder, parameterSheet, initialStatesPath, 
                         baseSeed=masterSeed, logQueue=logQueue, networkNumber=networkNumber)
    test_pair = pairs[startIndex:]
    
    try:
        active_processes = {}
        next_param = 0
        logQueue.put("Starting main processing loop")

        while next_param < len(test_pair) or active_processes:
            logQueue.put(f"Current loop state: next_param={next_param}, active_processes={len(active_processes)}")
            
            while len(active_processes) < 6 and next_param < len(test_pair):
                logQueue.put(f"Starting parameter {next_param}")
                executor = ProcessPoolExecutor(max_workers=1)
                future = executor.submit(runSimulation, test_pair[next_param] + (logQueue,))
                active_processes[future] = (executor, next_param)
                next_param += 1

            logQueue.put(f"Waiting for completion. Active processes: {len(active_processes)}")
            done, _ = wait(active_processes.keys(), return_when=FIRST_COMPLETED)
            
            for future in done:
                executor, param_idx = active_processes[future]
                try:
                    logQueue.put(f"Processing results for parameter {param_idx}")
                    future.result()
                except Exception as exc:
                    print(f"Error in parameter {param_idx}: {exc}")
                    logQueue.put(f"ERROR in parameter {param_idx}")
                    logException(exc, logQueue)
                finally:
                    logQueue.put(f"Cleaning up parameter {param_idx}")
                    executor.shutdown()
                    del active_processes[future]
                    gc.collect()
                    if hasattr(os, 'malloc_trim'):
                        os.malloc_trim(0)
                    logQueue.put(f"Cleanup complete for parameter {param_idx}")
                
    finally:
        logQueue.put("Performing final cleanup")
        for future, (executor, param_idx) in active_processes.items():
            try:
                executor.shutdown()
            except:
                pass
        gc.collect()
        logQueue.put(None)
        listener.join()
################

if __name__ == "__main__":
    masterSeed = 147208
    startIndex = int(sys.argv[1])
    networkNumber = int(sys.argv[2])
    reactionsFolder = f"/home/mzo5929/Keerthana/GRNsimulation/highthroughputData/linearNetworksSimulationSetup/moreLinearNetworkSimulationSetup/graph{networkNumber}/reactions/"
    # parameterSheet = "/home/mzo5929/Keerthana/GRNsimulation/highthroughputData/parameterSets/logRanges2020Parameters10500Seed42_August14.csv"
    parameterSheet = "/home/mzo5929/Keerthana/GRNsimulation/highthroughputData/parameterSets/parameterSet_26Nov2024.csv"
    initialStatesPath = f"/home/mzo5929/Keerthana/GRNsimulation/highthroughputData/linearNetworksSimulationSetup/moreLinearNetworkSimulationSetup/graph{networkNumber}/initialStates.csv"
    main(reactionsFolder, parameterSheet, initialStatesPath, masterSeed, startIndex, networkNumber)
