from simulationPerSet import createSimulation
import os
import pandas as pd
import numpy as np
from logging import handlers
from multiprocessing.queues import Queue
import logging
from logging.handlers import QueueHandler, RotatingFileHandler
import datetime
from numpy.random import default_rng
import multiprocessing 
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import List, Tuple
from time import sleep
from  multiprocessing.queues import Queue
import logging
from logging.handlers import QueueHandler, RotatingFileHandler
from multiprocessing.managers import SyncManager
import traceback
import sys
import gc
import psutil
import gc


def startManager():
    m = SyncManager()
    m.start()
    return m


def logException(exc, logQueue):
    # Get the full traceback as a string
    traceback_str = ''.join(
        traceback.format_exception(None, exc, exc.__traceback__))

    # Log the exception with the full traceback
    logQueue.put(f"A task generated an exception: {exc}\n{traceback_str}")


def setupLogging(queue):
    # Create a logger
    logger = logging.getLogger('my_logger')
    # Set the minimum level of logs for the logger
    logger.setLevel(logging.DEBUG)

    # Setup QueueHandler to send logs to the queue
    qh = QueueHandler(queue)
    logger.addHandler(qh)


def logListenerProcess(queue):
    # Setup the logging handler and formatter only once outside the loop
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")
    logFolder = "/home/mzo5929/Keerthana/Kuznets-Speck_PerturbationAnalysis/logs/"
    os.makedirs(logFolder, exist_ok = True)
    handler = RotatingFileHandler(
        f'{logFolder}/Logfile_{formatted_datetime}.log', maxBytes=1048576, backupCount=40)
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

def readInputData(reactionsPath: str, parameterSheet: str, initialStatesPath: str, logQueue: Queue, baseSeed: int = 101010, networkNumber: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame, List[int], pd.DataFrame]]:
    """
    Reads reaction data, parameters, and initial states, and generates random seeds for simulations.

    Args:
        reactionsPath (str): Path to the file containing the reactions.
        parameterSheet (str): Path to the CSV file containing parameters.
        initialStatesPath (str): Path to the CSV file containing initial states.
        baseSeed (int): Seed for the random number generator.

    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame, List[int], pd.DataFrame]]: List of tuples containing 
            reaction data, parameters, seeds, and initial states for simulations.
    """
    rng = default_rng(baseSeed)
    logQueue.put("Creating Input Pairs")
    reactionFiles = [f"{reactionsFile}"]
    if reactionFiles:
        reactionFile = reactionFiles  # Safely access the first file
    else:
        logQueue.put("No CSV files found in the directory")
        return []  # Return an empty list or handle this case appropriately
    reactionFiles.sort()
    parameterData = pd.read_csv(parameterSheet, index_col=0)
    initialStates = pd.read_csv(initialStatesPath, index_col=False)
    pairs = []
    numSeedsPerCombination = 3
    seedsAll = rng.integers(low=0,
                            high=2**32,
                            size=numSeedsPerCombination * len(reactionFiles) *
                            parameterData.shape[0]).tolist()
    index = 0
    for i, reactionFile in enumerate(reactionFiles):
        reactionDataFrame = pd.read_csv(reactionFile)
        for paramRow, parameter in parameterData.iterrows():
            if (paramRow <3):
                singleRowDf = parameter.to_frame(
                ).T  # Transpose to make it a single row DataFrame
                # fraction = singleRowDf['fraction'].iloc[0]
                # populationFraction = singleRowDf['populationFraction'].iloc[0]
                columnsToMelt = [
                    col for col in singleRowDf.columns
                    # if col not in ['fraction', 'populationFraction']
                ]
                meltedParameterDf = singleRowDf[columnsToMelt].melt(
                    var_name='parameter', value_name='value')
                # formula = "(0.5 * {d} * ({rProd}/{rDeg}))"
                # newRow = pd.DataFrame({
                # 'parameter': ['k'],
                # 'value': [formula]
                # })
                # meltedParameterDf = pd.concat([newRow,meltedParameterDf], ignore_index=True)
                startIndex = (i * len(parameterData) +
                              index) * numSeedsPerCombination
                # print(paramRow)
                seeds = seedsAll[startIndex:startIndex +
                                 numSeedsPerCombination]
                pairs.append(
                    (reactionDataFrame,
                     meltedParameterDf, seeds, initialStates, paramRow,
                     str(networkNumber)))
                index = index + 1

    del seedsAll
    del parameterData
    gc.collect()
    pairs.sort(key=lambda x: x[4])
    logQueue.put("Created Input Pairs")
    return pairs

def runSimulation(pair: Tuple[pd.DataFrame, pd.Series, List[int], pd.DataFrame, Queue, str, int]) -> None:
    """
    Runs a simulation based on provided reaction data, parameters, and initial states.

    Args:
        pair (Tuple[pd.DataFrame, pd.Series, List[int], pd.DataFrame]): Tuple containing reaction data,
            parameters, seeds, and initial states.
    """
    reaction, parameter, seeds, initialStates, parameterRow, graphName, logQueue, speciesToPerturb, timeOfPerturbation, amountToPerturb = pair
    createSimulation(
        initialStates,
        reactions=reaction,
        propensityParameters=parameter,
        steadyStateTime=2000,
        numThreads=10,
        seed=seeds,
        initialTime=100,
        postSplitTime=2000,
        nCells=10000,
        outputPath=f"/home/mzo5929/Keerthana/Kuznets-Speck_PerturbationAnalysis/rawData/simulationSet1/simulations/differentLevelsPerturbationNew/",
        nSims=2000,
        parameterRow = parameterRow,
        graphName = graphName, 
        logQueue = logQueue,
        speciesToPerturb = speciesToPerturb,
        timeOfPerturbation=timeOfPerturbation,
        amountToPerturb = amountToPerturb
    )


def main(reactionsFolder: str, parameterSheet: str, initialStatesPath: str, masterSeed: int, startIndex: int, networkNumber: int, speciesToPerturb: str = None, timeToPerturb: int = None, amountToPerturb: float = None):
    """
    Main function to process files and run simulations.

    Args:
        reactionsFolder (str): Path to reactions folder.
        parameterSheet (str): Path to parameter sheet CSV.
        initialStatesPath (str): Path to initial states CSV.
        masterSeed (int): Master seed for RNG.
    """

    manager = startManager()
    logQueue = manager.Queue()
    listener = multiprocessing.Process(
        target=logListenerProcess, args=(logQueue,))
    listener.start()

    pairs = readInputData(reactionsFolder, parameterSheet, initialStatesPath,
                          baseSeed=masterSeed, logQueue=logQueue, networkNumber=networkNumber)
    test_pair = pairs[:1]

    try:
        active_processes = {}
        next_param = 0
        logQueue.put("Starting main processing loop")

        while next_param < len(test_pair) or active_processes:
            logQueue.put(
                f"Current loop state: next_param={next_param}, active_processes={len(active_processes)}"
            )

            while len(active_processes) < 1 and next_param < len(test_pair):
                logQueue.put(f"Starting parameter {next_param + startIndex}")
                executor = ProcessPoolExecutor(max_workers=1)
                future = executor.submit(runSimulation,
                                         test_pair[next_param] + (logQueue, speciesToPerturb, timeToPerturb,amountToPerturb, ))
                active_processes[future] = (executor, next_param)
                next_param += 1

            logQueue.put(
                f"Waiting for completion. Active processes: {len(active_processes)}"
            )
            done, _ = wait(active_processes.keys(),
                           return_when=FIRST_COMPLETED)

            for future in done:
                executor, param_idx = active_processes[future]
                try:
                    logQueue.put(
                        f"Processing results for parameter {param_idx}")
                    future.result()
                except Exception as exc:
                    print(f"Error in parameter {param_idx}: {exc}")
                    logQueue.put(f"ERROR in parameter {param_idx}")
                    logException(exc, logQueue)
                finally:
                    del active_processes[future]
                    gc.collect()
                    if hasattr(os, 'malloc_trim'):
                        os.malloc_trim(0)

    finally:
        for future, (executor, param_idx) in active_processes.items():
            try:
                executor.shutdown()
            except:
                pass
        gc.collect()
        logQueue.put(None)
        listener.join()


if __name__ == "__main__":
    masterSeed = 147208
    startIndex = int(sys.argv[1])
    networkNumber = int(sys.argv[2])
    if len(sys.argv) > 4:
        geneToPerturb = str(sys.argv[3])
        timeOfPerturbation = int(sys.argv[4])
    else:
        geneToPerturb = []
        timeOfPerturbation = np.inf
    amountToPerturbList = np.arange(1.9, 4.1, 0.2)
    for amountToPerturb in amountToPerturbList:
        rawFilesFolder = "/home/mzo5929/Keerthana/Kuznets-Speck_PerturbationAnalysis/rawData/simulationSet1/setupData/"
        reactionsFile = f"{rawFilesFolder}/reactions.csv"
        parameterSheet = f"{rawFilesFolder}/parameterSet_26Nov2024.csv"
        initialStatesPath = f"{rawFilesFolder}/initialStates.csv"
        main(reactionsFile, parameterSheet, initialStatesPath, masterSeed, startIndex,
            networkNumber, speciesToPerturb=geneToPerturb, timeToPerturb=timeOfPerturbation, amountToPerturb=amountToPerturb)
