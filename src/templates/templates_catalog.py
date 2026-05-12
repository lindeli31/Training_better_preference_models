from typing import Callable

from core.Template import Template
from templates.HumanTemplate import HumanTemplate

templates_catalog: dict[str, Template] = {
    "human_template": HumanTemplate(),
}