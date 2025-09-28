from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List

router = APIRouter(
    prefix="/api/v1",
    tags=["model"],
    responses={404: {"description": "Not found"}},
)

class ReviewRequest(BaseModel):
    text: str
    
class ClassificationResponse(BaseModel):
    predictions: Dict[str, float]
    explanation: Dict[str, List[Dict[str, float]]]

@router.get("/health")
async def health_check():
    """Check if the model is healthy and ready to serve predictions"""
    return {"status": "ok", "model": "active"}

# Additional model-related endpoints can be added here