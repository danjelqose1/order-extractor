from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class Row(BaseModel):
    order_number: str = Field(default="", description="Order number like R-25-0716; leave blank if unknown")
    type: str = Field(..., description="Full glass type line as seen")
    dimension: str = Field(
    ...,
    pattern=r"^(?:\d{2,4}x\d{2,4})?$",
    description="Dimension in mm, WIDTHxHEIGHT, no spaces; leave empty ('') if not present"
)
    position: str = Field(..., description="Position code as seen")
    quantity: int = Field(ge=1, default=1)
    area: float = Field(ge=0.0, description="Area in square meters")

class ExtractionResult(BaseModel):
    order_number: str = Field(default="", description="Main order number detected")
    rows: List[Row]
    warnings: Optional[List[str]] = None

    @field_validator("rows")
    @classmethod
    def ensure_order_number(cls, v, info):
        main = info.data.get("order_number","")
        for row in v:
            if not row.order_number and main:
                row.order_number = main
        return v
