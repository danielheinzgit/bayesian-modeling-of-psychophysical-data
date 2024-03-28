from pathlib import Path
from typing import Union
from scipy.stats import norm
import numpy as np

def get_top_directory() -> Path:
    """
    Find the path to the project's top-level directory.

    Depends on having the local package installed and this 
    file in the source directory (e.g. with `pip install -e`).

    Returns:
    The path to the top-level directory
    """

    # use the location of the installed local package to figure out our absolute directory:
    tmp_path = Path(__file__).parent.parent.absolute()
    # assumes that we're running from a file in a subdirectory of the top dir.
    if (tmp_path / "tests").exists():
        # we've found the correct directory:
        return tmp_path
    else:
        raise ValueError(f"Couldn't find correct project directory; found {tmp_path}.")

def compute_dprime(hit_rate : Union[float, np.ndarray], false_alarm_rate : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the sensitivity (d') based on hit rate and false alarm rate.

    Parameters:
        hit_rate: Union[float, np.ndarray]
            The hit rate for target-present trials.
        false_alarm_rate: Union[float, np.ndarray]
            The false alarm rate for target-absent trials.

    Returns:
        Union[float, np.ndarray]
            The sensitivity (d').
    """
    return norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)

def compute_criterion(false_alarm_rate : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the criterion (lambda) based on false alarm rate.

    Parameters:
        false_alarm_rate: Union[float, np.ndarray]
            The false alarm rate for target-absent trials.

    Returns:
        Union[float, np.ndarray]
            The criterion (lambda).
    """
    return -norm.ppf(false_alarm_rate)