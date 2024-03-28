import pytest
import numpy as np
from scipy.stats import norm
from hypothesis import strategies as st
from hypothesis import settings

from .src import compute_dprime, compute_criterion


@pytest.mark.parametrize("hit_rate", st.floats(min_value=0.0, max_value=1.0))
@pytest.mark.parametrize("false_alarm_rate", st.floats(min_value=0.0, max_value=1.0))
def test_compute_dprime(hit_rate: float, false_alarm_rate: float) -> None:
    assert np.allclose(
        compute_dprime(hit_rate, false_alarm_rate),
        np.round(norm.ppf(hit_rate) - norm.ppf(false_alarm_rate), 4)
    )

@pytest.mark.parametrize("false_alarm_rate", st.floats(min_value=0.0, max_value=1.0))
def test_compute_criterion(false_alarm_rate: float) -> None:
    assert np.allclose(
        compute_criterion(false_alarm_rate),
        -np.round(norm.ppf(false_alarm_rate), 4)
    )

@pytest.mark.parametrize("hit_rate", st.arrays(min_size=10, elements=st.floats(min_value=0.0, max_value=1.0)))
@pytest.mark.parametrize("false_alarm_rate", st.arrays(min_size=10, elements=st.floats(min_value=0.0, max_value=1.0)))
def test_compute_dprime_array(hit_rate: np.ndarray, false_alarm_rate: np.ndarray) -> None:
    result = compute_dprime(hit_rate, false_alarm_rate)
    assert np.allclose(result, np.round(norm.ppf(hit_rate) - norm.ppf(false_alarm_rate), 4))

settings.register_profile('fast check', deadline=None)
settings.load_profile('fast check')