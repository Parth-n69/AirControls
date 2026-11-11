"""Pytest configuration and synthetic hand landmark fixtures for AirControl tests."""

import pytest
import numpy as np
from aircontrol.utils.synthetic import create_synthetic_hand


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def random_noise_frame():
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
