"""
    This module contains the paths to the root of the project, the root of the test folder, and the root of the data folder.
"""
import os
from pathlib import Path

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_PATH_TEST_DATA = Path("tests/data/")
