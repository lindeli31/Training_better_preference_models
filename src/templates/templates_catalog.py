from typing import Callable

from src.core.Template import Template
from src.templates.GepaApertus8bTemplate import GepaApertus8bTemplate
from src.templates.HumanTemplate import HumanTemplate
from src.templates.Mipro2Gemma4Template import Mipro2Gemma4Template
from src.templates.Mipro4Gemma4Template import Mipro4Gemma4Template
from src.templates.MiproGemma4Template import MiproGemma4Template

templates_catalog: dict[str, Template] = {
    "human_template": HumanTemplate(),
    "gepa_apertus8b_template": GepaApertus8bTemplate(),
    "mipro_gemma4_template": MiproGemma4Template(),
    "mipro2_gemma4_template": Mipro2Gemma4Template(),
    "mipro4_gemma4_template": Mipro4Gemma4Template(),
}