"""public.degradation swaps the version and cache directory, then restores them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import data_io
import public


def test_switch_restores():
    before = config.DEGRADE, data_io.PUBLIC_CACHE
    with public.degradation(2):
        assert config.DEGRADE == 2
        assert data_io.PUBLIC_CACHE.name == "cache-v2"
    assert (config.DEGRADE, data_io.PUBLIC_CACHE) == before
