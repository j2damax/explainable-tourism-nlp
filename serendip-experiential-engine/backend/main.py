from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import Dict, List
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Serendip Experiential Engine API",
    description="API for classifying experiential dimensions in Sri Lankan tourism reviews",
    version="0.1.0"
)

class ReviewRequest(BaseModel):
    text: str

class ExplanationItem(BaseModel):
    word: str
    value: float
    
class ClassificationResponse(BaseModel):
    predictions: Dict[str, float]
    explanation: Dict[str, List[ExplanationItem]]

# Define the experiential dimensions
DIMENSIONS = [
    "Regenerative & Eco-Tourism",
    "Integrated Wellness",
    "Immersive Culinary",
    "Off-the-Beaten-Path Adventure"
]

@app.get("/")
def read_root():
    """Root endpoint for health checking"""
    return {"status": "active", "service": "Serendip Experiential Engine API"}

@app.get("/dimensions")
def get_dimensions():
    """Get all available experiential dimensions"""
    return {"dimensions": DIMENSIONS}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_review(request: ReviewRequest):
    """
    Classify a tourism review into experiential dimensions
    """
    try:
        logger.info(f"Processing review: {request.text[:50]}...")
        
        # TODO: Replace this with actual model inference
        # This is just placeholder logic that returns random values
        mock_predictions = {
            dim: float(np.random.random()) for dim in DIMENSIONS
        }
        
        # Mock explanation data (in a real app, this would come from SHAP or similar)
        mock_explanation = {
            dim: [
                {"word": "beautiful", "value": float(np.random.random())},
                {"word": "amazing", "value": float(np.random.random())},
                {"word": "sustainable", "value": float(np.random.random())}
            ] for dim in DIMENSIONS
        }
        
        return {
            "predictions": mock_predictions,
            "explanation": mock_explanation
        }
    
    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing review: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Determine the host and port from environment variables or use defaults
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application
    uvicorn.run("main:app", host=host, port=port, reload=True)