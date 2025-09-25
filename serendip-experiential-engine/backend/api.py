"""
API module for BertForSequenceClassification model loading and inference
"""
import os
from typing import List, Dict, Any, Optional
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import shap
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model constants
MODEL_NAME = "j2damax/serendip-travel-classifier"
NUM_LABELS = 4
MAX_LENGTH = 512

# Dimension labels
DIMENSIONS = [
    "Regenerative & Eco-Tourism",
    "Integrated Wellness",
    "Immersive Culinary",
    "Off-the-Beaten-Path Adventure"
]

# Initialize FastAPI app
app = FastAPI(
    title="Serendip Travel Classifier API",
    description="API for classifying experiential dimensions in Sri Lankan tourism reviews using BERT",
    version="0.1.0",
)

# Request and response models
class PredictRequest(BaseModel):
    review_text: str

class PredictionResult(BaseModel):
    label: str
    score: float

class ExplainRequest(BaseModel):
    review_text: str
    top_n_words: int = 10

class ExplainResponse(BaseModel):
    html: str
    top_words: Dict[str, List[Dict[str, Any]]]

# Global variables for model, tokenizer, and classifier
model = None
tokenizer = None
classifier = None
shap_explainer = None

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer from Hugging Face Hub on startup"""
    global model, tokenizer, classifier
    
    try:
        logger.info(f"Loading model {MODEL_NAME}...")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification"
        )
        
        # Create classification pipeline
        classifier = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer,
            function_to_apply="sigmoid",
            top_k=None,
            device=-1 if not torch.cuda.is_available() else 0
        )
        
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # We'll continue anyway and try to load the model on the first request if needed

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"status": "active", "model": MODEL_NAME}

@app.post("/predict", response_model=List[PredictionResult])
async def predict(request: PredictRequest):
    """
    Classify a tourism review into experiential dimensions
    
    This endpoint processes the review text and returns prediction scores for all dimensions.
    """
    global model, tokenizer, classifier
    
    # Ensure model is loaded
    if classifier is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, 
                num_labels=NUM_LABELS,
                problem_type="multi_label_classification"
            )
            classifier = pipeline(
                "text-classification", 
                model=model, 
                tokenizer=tokenizer,
                function_to_apply="sigmoid", 
                top_k=None,
                device=-1 if not torch.cuda.is_available() else 0
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    try:
        logger.info(f"Processing review: {request.review_text[:50]}...")
        
        # Run inference
        result = classifier(request.review_text)
        
        # Extract predictions and format response
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            # Handle the case where the pipeline returns a list with one dict
            scores = result[0]
            
            # Format output as a list of label-score pairs
            formatted_results = []
            
            for idx, label in enumerate(DIMENSIONS):
                label_id = f"LABEL_{idx}"
                score = scores.get(label_id, 0.0)
                if isinstance(score, torch.Tensor):
                    score = score.item()
                formatted_results.append({
                    "label": label,
                    "score": float(score)
                })
                
            # Sort by score in descending order
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            return formatted_results
        else:
            # If the pipeline returns something unexpected, try to convert it
            formatted_results = []
            for i, label in enumerate(DIMENSIONS):
                score = 0.0
                if i < len(result):
                    if isinstance(result[i], dict) and "score" in result[i]:
                        score = result[i]["score"]
                    elif isinstance(result[i], (int, float, np.number, torch.Tensor)):
                        score = float(result[i])
                
                formatted_results.append({
                    "label": label, 
                    "score": float(score)
                })
                
            return formatted_results
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Generate SHAP explanations for a review's classification
    
    This endpoint returns both HTML visualization and top influencing words.
    """
    global model, tokenizer, classifier, shap_explainer
    
    # Check if model is loaded
    if model is None or tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, 
                num_labels=NUM_LABELS,
                problem_type="multi_label_classification"
            )
            classifier = pipeline(
                "text-classification", 
                model=model, 
                tokenizer=tokenizer,
                function_to_apply="sigmoid", 
                top_k=None,
                device=-1 if not torch.cuda.is_available() else 0
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    try:
        # Tokenize the input
        review_text = request.review_text
        inputs = tokenizer(
            review_text, 
            return_tensors="pt",
            truncation=True, 
            padding=True, 
            max_length=MAX_LENGTH
        )
        
        # Create a simplified tokenize function for SHAP
        def model_predict(inputs_text):
            inputs_encoded = tokenizer(
                inputs_text, 
                return_tensors="pt",
                truncation=True, 
                padding=True, 
                max_length=MAX_LENGTH
            )
            with torch.no_grad():
                outputs = model(**inputs_encoded)
                predictions = torch.sigmoid(outputs.logits)
            return predictions.detach().numpy()
        
        # Create a SHAP explainer
        # For this example, we'll use the Partition explainer which is more suitable for text data
        if shap_explainer is None:
            shap_explainer = shap.Explainer(
                model_predict, 
                tokenizer=lambda x: x.split()
            )
        
        # Get SHAP values
        shap_values = shap_explainer([review_text])
        
        # Create visualization HTML
        words = review_text.split()
        shap_values_for_text = shap_values[0]
        
        # Generate SHAP force plot HTML
        force_plot_html = shap.plots.force(
            shap_values_for_text[0], 
            feature_names=words,
            matplotlib=False,
            show=False,
            out_names=DIMENSIONS
        )
        
        # Get top words for each dimension
        top_words = {}
        for i, dimension in enumerate(DIMENSIONS):
            # Extract SHAP values for this dimension
            dim_shap_values = shap_values_for_text[0, :, i]
            
            # Get top N words with highest absolute SHAP values
            top_indices = np.argsort(np.abs(dim_shap_values))[-request.top_n_words:][::-1]
            top_words_for_dim = []
            
            for idx in top_indices:
                if idx < len(words):
                    top_words_for_dim.append({
                        "word": words[idx],
                        "value": float(dim_shap_values[idx]),
                        "is_positive": bool(dim_shap_values[idx] > 0)
                    })
            
            top_words[dimension] = top_words_for_dim
        
        return {
            "html": force_plot_html,
            "top_words": top_words
        }
        
    except Exception as e:
        logger.error(f"Error during explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Determine the host and port from environment variables or use defaults
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application
    uvicorn.run("api:app", host=host, port=port, reload=True)