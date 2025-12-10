import os
import pytest
import jax.numpy as jnp
import sys

from diffrax import SaveAt
from pathlib import Path

current_dir = Path(os.getcwd())
sys.path.append(str(current_dir))

from src.model_training.training_setup import setup_curriculum_learning


@pytest.fixture
def dummy_data():
    return jnp.linspace(0, 1, 10)

@pytest.fixture
def dummy_t_save():
    return SaveAt(
        ts=jnp.linspace(0.0, 1.0, 10)
    )

def test_returns_none_when_curriculum_learning_disabled_single(dummy_data, dummy_t_save):
    result = setup_curriculum_learning(
        data=dummy_data,
        t_save=dummy_t_save,
        n_epoch=100,
        shooting_mode="single",
        shooting_window_length=3,
        curriculum_learning=False,
        num_curriculum_steps=5,
        max_curriculum_epochs=10
    )
    assert result is None


def test_returns_none_when_curriculum_learning_disabled_multiple(dummy_data, dummy_t_save):
    result = setup_curriculum_learning(
        data=dummy_data,
        t_save=dummy_t_save,
        n_epoch=100,
        shooting_mode="multiple",
        shooting_window_length=3,
        curriculum_learning=False,
        num_curriculum_steps=5,
        max_curriculum_epochs=10
    )
    assert result is None

def test_single_shooting_curriculum_structure(dummy_data, dummy_t_save):
    stage_dict = setup_curriculum_learning(
        data=dummy_data,
        t_save=dummy_t_save,
        n_epoch=100,
        shooting_mode="single",
        shooting_window_length=3,
        curriculum_learning=True,
        num_curriculum_steps=5,
        max_curriculum_epochs=50
    )
    assert isinstance(stage_dict, dict)
    assert len(stage_dict) == 5
    for stage in stage_dict.values():
        assert "start" in stage
        assert "end" in stage
        assert "length" in stage
        assert "time" in stage
        assert "mask" in stage
        assert "last_valid_index" in stage

def test_multiple_shooting_curriculum_structure(dummy_data, dummy_t_save):
    stage_dict = setup_curriculum_learning(
        data=dummy_data,
        t_save=dummy_t_save,
        n_epoch=100,
        shooting_mode="multiple",
        shooting_window_length=4,
        curriculum_learning=True,
        num_curriculum_steps=99,  # should be ignored
        max_curriculum_epochs=50
    )
    assert isinstance(stage_dict, dict)
    # Based on shooting_window_length=4 and len(data)=10, expect ceil(10/3) = 4 windows
    assert len(stage_dict) == 4
    for stage in stage_dict.values():
        assert "start" in stage
        assert "end" in stage
        assert "length" in stage
        assert "time" in stage
        assert "mask" in stage
        assert "last_valid_index" in stage

def test_error_on_invalid_shooting_mode(dummy_data, dummy_t_save):
    with pytest.raises(ValueError, match="shooting_mode"):
        setup_curriculum_learning(
            data=dummy_data,
            t_save=dummy_t_save,
            n_epoch=100,
            shooting_mode="invalid_mode",
            shooting_window_length=3,
            curriculum_learning=True,
            num_curriculum_steps=5,
            max_curriculum_epochs=10
        )

def test_error_on_missing_max_curriculum_epochs(dummy_data, dummy_t_save):
    with pytest.raises(ValueError, match="Max curriculum steps"):
        setup_curriculum_learning(
            data=dummy_data,
            t_save=dummy_t_save,
            n_epoch=100,
            shooting_mode="single",
            shooting_window_length=3,
            curriculum_learning=True,
            num_curriculum_steps=5,
            max_curriculum_epochs=None
        )

def test_error_on_invalid_num_curriculum_steps(dummy_data, dummy_t_save):
    with pytest.raises(ValueError, match="Number of steps must be positive"):
        setup_curriculum_learning(
            data=dummy_data,
            t_save=dummy_t_save,
            n_epoch=100,
            shooting_mode="single",
            shooting_window_length=3,
            curriculum_learning=True,
            num_curriculum_steps=0,
            max_curriculum_epochs=20
        )
