"""Pytest configuration and fixtures."""

import pytest
import torch

from bbpm import set_global_seed


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment with fixed seed."""
    set_global_seed(42)
    yield
    # Cleanup if needed
