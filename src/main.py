# src/main.py

from fastapi import FastAPI, BackgroundTasks
import src.initialize  # Ensure Constants is set
from .constants import Constants
import subprocess
from pydantic import BaseModel

app = FastAPI()

class TrainRequest(BaseModel):
    pass  # No parameters for now; can extend later

class TrainResponse(BaseModel):
    message: str

def run_pipeline():
    subprocess.run(["dvc", "repro"], check=True)

@app.post("/train", response_model=TrainResponse)
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_pipeline)
    return {"message": "Pipeline started in the background"}

# Critical Comment: Removed direct config loading since Constants is set via src/initialize.py.
# The hardcoded YAML paths are now accessed via Constants.CONFIG_PATH and Constants.HYPERPARAMS_PATH if needed,
# though not used here as the pipeline is triggered via DVC.