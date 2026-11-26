"""YAML config <-> defaults, overrides, validation."""
import os

import pytest

from crossing_target_tracking.core.config import Config, load_config, to_dict

YAML = os.path.join(os.path.dirname(__file__), '..', 'config', 'crossing_tracking.yaml')


def test_yaml_matches_dataclass_defaults():
    assert to_dict(load_config(YAML)) == to_dict(Config())


def test_overrides_and_validation():
    c = load_config(YAML, ['shadow.enabled=false', 'high.manager.delete_n=20', 'grid.normalization=raw_count'])
    assert c.shadow.enabled is False and c.high.manager.delete_n == 20
    with pytest.raises(ValueError):
        load_config(YAML, ['grid.nonexistent=1'])
    with pytest.raises(ValueError):
        load_config(YAML, ['registration.mode=magic'])
    with pytest.raises(ValueError):
        load_config(YAML, ['low_manager.confirm_m=9'])
