import os
import random

from greekkv.utils import seed_all


def test_seed_all_makes_random_deterministic():
    seed_all(1234)
    a = [random.random() for _ in range(5)]
    seed_all(1234)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_all_sets_pythonhashseed():
    seed_all(7)
    assert os.environ["PYTHONHASHSEED"] == "7"
