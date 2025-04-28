## Utility Scripts Reference

### 1. `findMissingParams.py`
- Checks for missing parameter files in a directory using a regex pattern.
- CLI example:
  ```bash
  python utilityModules/findMissingParams.py --directory /path/to/data --pattern 'parameterRow(\d+)_.*\.csv$'
- can be used in a python script or notebook:
  ```python
  from utilityModules.findMissingParams import summarizeMissing

  existing, missing = summarizeMissing(
      directory="/path/to/data",
      pattern=r'parameterRow(\d+)_.*\.csv$',
      start=0,
      end=10500
  )
  print(f"Found {len(existing)} files, missing {len(missing)}")

### 2. 