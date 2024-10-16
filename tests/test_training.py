import numpy as np
import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from src.model_training.training_util import MetricHandler

@pytest.mark.parametrize("loss_values, exp_status, exp_counter",
                         [
                             ([np.nan for _ in range(5)], True, 5),
                             ([np.inf for _ in range(5)], True, 5),
                             ([np.nan, np.nan], False, 2),
                             ([np.nan, np.nan, np.nan, np.nan, 1.0], False, 0),
                         ])
def test_invalid_metric_handler(loss_values, exp_status, exp_counter):
    metric_handler = MetricHandler(patience=5)

    for loss in loss_values:
        status = metric_handler.check_whether_should_break(loss)
    counter = metric_handler.counter

    assert status == exp_status
    assert counter == exp_counter
