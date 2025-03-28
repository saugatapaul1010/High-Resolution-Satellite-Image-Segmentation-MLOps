# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import training, data, health
from api.config import settings

app = FastAPI(
    title="Satellite Segmentation MLOps API",
    description="API for satellite image segmentation MLOps pipeline",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(training.router, prefix="/training", tags=["Training"])

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Satellite Segmentation MLOps API",
        "docs_url": "/docs",
        "version": "0.1.0"
    }