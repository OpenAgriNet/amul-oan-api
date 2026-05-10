from pydantic import BaseModel, Field
from typing import Optional, List, Any

class BaseResponse(BaseModel):
    status: str = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Response message")

class ErrorResponse(BaseResponse):
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Any] = Field(None, description="Additional error details") 