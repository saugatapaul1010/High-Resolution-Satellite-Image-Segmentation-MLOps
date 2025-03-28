# api/services/data.py
import os
import shutil
import uuid
from datetime import datetime
from fastapi import Depends, UploadFile, BackgroundTasks, HTTPException
import subprocess
from typing import List, Dict, Any

from api.config import settings
from api.models.data import DatasetInfo, DatasetType, DataUploadResponse, DataProcessingResponse
from tasks.data_tasks import process_dataset

class DataService:
    """Service for data management operations."""
    
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        # Ensure directories exist
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List all available datasets."""
        datasets = []
        
        # Check raw datasets
        raw_dir = os.path.join(self.data_dir, "raw")
        for dataset in os.listdir(raw_dir):
            dataset_path = os.path.join(raw_dir, dataset)
            if os.path.isdir(dataset_path):
                # Count files and calculate size
                file_count = 0
                size_bytes = 0
                for root, _, files in os.walk(dataset_path):
                    file_count += len(files)
                    size_bytes += sum(os.path.getsize(os.path.join(root, file)) for file in files)
                
                # Get creation time
                created_at = datetime.fromtimestamp(os.path.getctime(dataset_path)).isoformat()
                
                datasets.append(DatasetInfo(
                    name=dataset,
                    type=DatasetType.RAW,
                    file_count=file_count,
                    size_bytes=size_bytes,
                    created_at=created_at
                ))
        
        # Check processed datasets
        processed_dir = os.path.join(self.data_dir, "processed")
        for dataset in os.listdir(processed_dir):
            dataset_path = os.path.join(processed_dir, dataset)
            if os.path.isdir(dataset_path):
                # Count files and calculate size
                file_count = 0
                size_bytes = 0
                for root, _, files in os.walk(dataset_path):
                    file_count += len(files)
                    size_bytes += sum(os.path.getsize(os.path.join(root, file)) for file in files)
                
                # Get creation time
                created_at = datetime.fromtimestamp(os.path.getctime(dataset_path)).isoformat()
                
                datasets.append(DatasetInfo(
                    name=dataset,
                    type=DatasetType.PROCESSED,
                    file_count=file_count,
                    size_bytes=size_bytes,
                    created_at=created_at
                ))
        
        return datasets
    
    async def upload_file(self, file: UploadFile) -> DataUploadResponse:
        """Upload satellite image or annotation data."""
        # Determine file type by extension
        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        extension = os.path.splitext(filename)[1].lower()
        
        # Validate file extension
        if extension not in [".png", ".jpg", ".jpeg", ".json"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file extension: {extension}. Supported: .png, .jpg, .jpeg, .json"
            )
        
        # Generate dataset name if JSON file (first file in a dataset)
        if extension == ".json":
            # Extract base name for dataset
            base_name = os.path.splitext(filename)[0]
            dataset_dir = os.path.join(self.data_dir, "raw", base_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Save file
            file_path = os.path.join(dataset_dir, filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Initialize DVC tracking if not already done
            self._init_dvc_tracking()
            
            return DataUploadResponse(
                filename=filename,
                size_bytes=os.path.getsize(file_path),
                status="success"
            )
        else:
            # Must match an existing dataset name for image files
            base_name = os.path.splitext(filename)[0]
            
            # Find matching dataset (same base name)
            matching_dirs = [
                d for d in os.listdir(os.path.join(self.data_dir, "raw"))
                if os.path.isdir(os.path.join(self.data_dir, "raw", d)) and d == base_name
            ]
            
            if not matching_dirs:
                raise HTTPException(
                    status_code=400,
                    detail=f"No dataset found for {base_name}. Upload JSON annotation file first."
                )
            
            # Save to matching dataset
            dataset_dir = os.path.join(self.data_dir, "raw", matching_dirs[0])
            file_path = os.path.join(dataset_dir, filename)
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            return DataUploadResponse(
                filename=filename,
                size_bytes=os.path.getsize(file_path),
                status="success"
            )
    
    def process_data(self, background_tasks: BackgroundTasks, dataset_name: str) -> DataProcessingResponse:
        """Process raw data into training/validation sets."""
        # Check if dataset exists
        dataset_dir = os.path.join(self.data_dir, "raw", dataset_name)
        if not os.path.exists(dataset_dir):
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
        
        # Launch background task for processing
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            process_dataset.delay,
            dataset_name,
            self.data_dir
        )
        
        return DataProcessingResponse(
            task_id=task_id,
            status="pending"
        )
    
    def list_data_versions(self) -> List[str]:
        """List all available data versions tracked by DVC."""
        try:
            # Ensure DVC is initialized
            self._init_dvc_tracking()
            
            # Get list of tags and commits
            result = subprocess.run(
                ["dvc", "list", "--recursive", "--dvc-only"], 
                capture_output=True, 
                text=True,
                check=True
            )
            
            # Parse output
            versions = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            return versions
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"DVC error: {e.stderr}")
    
    def checkout_data_version(self, version: str) -> Dict[str, Any]:
        """Checkout a specific data version."""
        try:
            # Checkout the specified version
            subprocess.run(
                ["dvc", "checkout", version],
                check=True
            )
            
            return {"status": "success", "message": f"Checked out version {version}"}
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"DVC checkout error: {e}")
    
    def _init_dvc_tracking(self):
        """Initialize DVC tracking if not already done."""
        # Check if DVC is initialized
        if not os.path.exists(os.path.join(self.data_dir, ".dvc")):
            try:
                # Initialize DVC repo
                subprocess.run(
                    ["dvc", "init", "--no-scm"],
                    cwd=self.data_dir,
                    check=True
                )
                
                # Configure default remote if specified
                if settings.DVC_REMOTE != "local":
                    subprocess.run(
                        ["dvc", "remote", "add", "default", settings.DVC_REMOTE],
                        cwd=self.data_dir,
                        check=True
                    )
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=f"Failed to initialize DVC: {e}")