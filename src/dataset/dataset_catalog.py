from typing import Callable

from datasets import Dataset

from dataset.HelpSteer2 import get_help_steer_2
from dataset.HelpSteer3 import get_help_steer_3

dataset_catalog: dict[str, Callable[[], Dataset]] = {
    "help_steer_2": get_help_steer_2,
    "help_steer_3": get_help_steer_3,
}