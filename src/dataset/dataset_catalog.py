from typing import Callable

from datasets import Dataset

from src.dataset.HelpSteer2 import get_help_steer_2
from src.dataset.HelpSteer3 import get_help_steer_3
from src.dataset.HelpSteer3NoHistory import get_help_steer_3_no_history

dataset_catalog: dict[str, Callable[[], Dataset]] = {
    "help_steer_2": get_help_steer_2,
    "help_steer_3": get_help_steer_3,
    "get_help_steer_3_no_history": get_help_steer_3_no_history,
}