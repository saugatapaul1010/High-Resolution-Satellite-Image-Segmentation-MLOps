# api/routers/data.py
from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, HTTPException
from typing import List

from api.models.data import DatasetInfo, DataUploadResponse, DataProcessingResponse
from api.services.data import DataService

router = APIRouter()

@router.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets(data_service: DataService = Depends()):
    """List all available datasets."""
    return data_service.list_datasets()

@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    data_service: DataService = Depends()
):
    """Upload satellite image or annotation data."""
    return await data_service.upload_file(file)

@router.post("/process", response_model=DataProcessingResponse)
async def process_data(
    background_tasks: BackgroundTasks,
    dataset_name: str,
    data_service: DataService = Depends()
):
    """Process raw data into training/validation sets."""
    return data_service.process_data(background_tasks, dataset_name)

@router.get("/versions", response_model=List[str])
async def list_data_versions(data_service: DataService = Depends()):
    """List all available data versions tracked by DVC."""
    return data_service.list_data_versions()

@router.post("/checkout/{version}", response_model=dict)
async def checkout_data_version(
    version: str,
    data_service: DataService = Depends()
):
    """Checkout a specific data version."""
    return data_service.checkout_data_version(version)