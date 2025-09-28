from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["model"],
    responses={404: {"description": "Not found"}},
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

@router.get("/health")
async def health_check():
    """Check if the model is healthy and ready to serve predictions"""
    return {"status": "ok", "model": "active"}

@router.get("/dimensions")
async def get_dimensions():
    """Get all available experiential dimensions"""
    return {"dimensions": DIMENSIONS}

@router.post("/classify", response_model=ClassificationResponse)
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