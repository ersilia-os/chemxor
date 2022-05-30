"""FHE Encrypted model API schema."""

from typing import Optional

from pydantic import BaseModel, Field


class FHEModelQueryPostRequest(BaseModel):
    """FHE Encrypted model query post request schema."""

    ts_context: str = Field(unicode_safe=False)
    input_tensor: str = Field(unicode_safe=False)


class FHEModelQueryPostResponse(BaseModel):
    """FHE Encrypted model query post response schema."""

    output_tensor: Optional[str] = Field(unicode_safe=False)
