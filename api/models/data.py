# api/models/data.py
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional

class DatasetType(str, Enum):
    RAW = "raw"
    PROCESSED = "processed"
    TRAIN = "train"
    VALIDATION = "validation"

class DatasetInfo(BaseModel):
    """Dataset information."""
    name: str
    type: DatasetType
    file_count: int
    size_bytes: int
    created_at: str
    
class DataUploadResponse(BaseModel):
    """Response for data upload endpoint."""
    filename: str
    size_bytes: int
    status: str
    
class DataProcessingResponse(BaseModel):
    """Response for data processing endpoint."""
    task_id: str
    status: str