"""FHE Encrypted model API schema."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class PreProcessInput(Enum):
    """Supported input pre processor."""

    IM_TO_COL = "im_to_col"
    RE_ENCRYPT = "re_encrypt"
    PASSTHROUGH = "passthrough"


class FHEModelQueryPostRequest(BaseModel):
    """FHE Encrypted model query post request schema."""

    ts_context: str = Field(unicode_safe=False)
    model_input: str = Field(unicode_safe=False)


class FHEModelQueryPostResponse(BaseModel):
    """FHE Encrypted model query post response schema."""

    model_output: Optional[str] = Field(unicode_safe=False)


class PartFHEModelQueryPostRequest(FHEModelQueryPostRequest):
    """Partitioned FHE Encrypted model query post request schema."""

    model_step: Optional[int]


class PartFHEModelQueryPostResponse(FHEModelQueryPostResponse):
    """Partitioned FHE Encrypted model query post response schema."""

    next_step: int
    preprocess_next_input: Optional[List[PreProcessInput]] = None
