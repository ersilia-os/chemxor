"""Test encryption schema."""

from pydantic import BaseModel, Field


class TestEncPost(BaseModel):
    """Test encryption post schema."""

    context: str = Field(unicode_safe=False)
    vector: str = Field(unicode_safe=False)


class TestEncResponse(BaseModel):
    """Test encryption response schema."""

    vector: str = Field(unicode_safe=False)
