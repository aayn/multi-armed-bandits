from bandit import Arm
import numpy as np


def test_type():
    "Assert that arm returns a float when called."
    a = Arm()
    assert type(a()) is np.float64 or type(a()) is float