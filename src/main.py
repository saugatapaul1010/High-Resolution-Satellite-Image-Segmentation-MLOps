from fastapi import FastAPI, BackgroundTasks
from src.config import load_config
import subprocess
from pydantic import BaseModel

app = FastAPI()
config = load_config('config/config.yaml', 'config/hyperparameters.yaml')

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