"""Test encryption schema."""

from typing import Any

from pydantic import BaseModel


class TestEncPost(BaseModel):
    """Test encryption post schema."""

    context: Any
    vector: Any


class TestEncResponse(BaseModel):
    """Test encryption response schema."""

    vector: Any
