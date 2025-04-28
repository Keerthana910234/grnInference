import os
import re
import argparse
from typing import Set, Tuple


def getParameterList(directory: str, pattern: str = r'parameter(\d+)\.csv$') -> Set[int]:
    """
    Extract unique parameter numbers from filenames in a given directory.

    Args:
        directory (str): Directory containing the files.
        pattern (str): Regex pattern to match filenames, with a capture group for parameter numbers.

    Returns:
        Set[int]: Set of unique parameter numbers found in matching filenames.
    """
    compiledPattern = re.compile(pattern)
    parameterSet = set()

    for filename in os.listdir(directory):
        match = compiledPattern.match(filename)
        if match:
            parameterNum = int(match.group(1))
            parameterSet.add(parameterNum)

    return parameterSet


def findMissingParameters(existingParams: Set[int], start: int = 0, end: int = 10500) -> Set[int]:
    """
    Find missing parameter numbers in a given numeric range.

    Args:
        existingParams (Set[int]): Set of existing parameter numbers.
        start (int): Start of expected parameter range (inclusive).
        end (int): End of expected parameter range (exclusive).

    Returns:
        Set[int]: Set of parameter numbers that are missing from existingParams.
    """
    return set(range(start, end)) - existingParams


def summarizeMissing(directory: str,
                     pattern: str = r'parameter(\d+)\.csv$',
                     start: int = 0,
                     end: int = 10500) -> Tuple[Set[int], Set[int]]:
    """
    Get sets of existing and missing parameter numbers based on files in a directory.

    Args:
        directory (str): Path to the directory containing parameter files.
        pattern (str): Regex pattern with a numeric capture group (default: r'parameter(\\d+)\\.csv$').
        start (int): Start of the parameter range (inclusive).
        end (int): End of the parameter range (exclusive).

    Returns:
        Tuple[Set[int], Set[int]]: A tuple containing (existingSet, missingSet).
    """
    existingSet = getParameterList(directory, pattern)
    missingSet = findMissingParameters(existingSet, start, end)
    return existingSet, missingSet


def main():
    """
    Command-line entry point for checking missing parameter files.

    Uses argparse to accept:
    --directory: Directory to search
    --pattern: Regex pattern with numeric capture group
    --start: Start of expected parameter range
    --end: End (exclusive) of expected parameter range
    """
    parser = argparse.ArgumentParser(description="Find missing parameter files in a directory.")
    parser.add_argument('--directory', required=True, help='Directory containing the files')
    parser.add_argument('--pattern', default=r'parameter(\d+)\.csv$', help='Regex pattern to match parameter files')
    parser.add_argument('--start', type=int, default=0, help='Start of parameter range')
    parser.add_argument('--end', type=int, default=10500, help='End (exclusive) of parameter range')

    args = parser.parse_args()

    existingParams, missingParams = summarizeMissing(args.directory, args.pattern, args.start, args.end)

    print(f"✅ Found {len(existingParams)} parameter files")
    print(f"❌ Missing {len(missingParams)} parameters")

    if missingParams:
        sortedMissing = sorted(missingParams)
        print(f"First few missing: {sortedMissing[:10]}")
        if len(sortedMissing) > 20:
            print(f"Last few missing: {sortedMissing[-10:]}")


if __name__ == "__main__":
    main()
