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

# Global variables for model, tokenizer, and classifier
model = None
tokenizer = None
classifier = None
shap_explainer = None

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer from Hugging Face Hub on startup"""
    global model, tokenizer, classifier
    
    logger.info(f"Starting API server. Model will be loaded on first request.")
    
    # We'll load the model on the first request to make startup faster
    # and avoid timeouts during container initialization
    pass

@app.get("/")
def read_root():
    """Health check endpoint"""
    global model, tokenizer, classifier
    
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "active", 
        "model": MODEL_NAME,
        "model_status": model_status
    }

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
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Device set to use {device}")
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
                device=-1 if device == "cpu" else 0
            )
            print("Model and classifier initialized successfully")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    try:
        logger.info(f"Processing review: {request.review_text[:50]}...")
        
        # Run inference
        result = classifier(request.review_text)
        
        # Print the raw result for debugging
        print(f"Raw prediction result: {result}")
        
        # Extract predictions and format response
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list) and len(result[0]) > 0:
                # Handle nested list structure [[{...}, {...}, ...]]
                predictions = result[0]
                formatted_results = []
                
                # Create a mapping from label names to scores
                label_scores = {item['label']: item['score'] for item in predictions}
                
                # Ensure we have results for all dimensions in the expected order
                for label in DIMENSIONS:
                    score = label_scores.get(label, 0.0)
                    if isinstance(score, torch.Tensor):
                        score = score.item()
                    formatted_results.append({
                        "label": label,
                        "score": float(score)
                    })
            elif isinstance(result[0], dict):
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

@app.post("/explain")
async def explain(request: ExplainRequest):
    """
    Generate explanations for a review's classification
    
    This endpoint returns both HTML visualization and top influencing words,
    using a simple attribution method for reliability.
    """
    global model, tokenizer, classifier
    
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
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    try:
        # Get the input text
        review_text = request.review_text
        
        # Tokenize the text by word (use simple space splitting for visualization)
        # NOTE: This is not the same as model tokenization, it's just for display
        words = review_text.split()
        if len(words) < 2:
            raise ValueError("Review text must contain at least 2 words for explanation")
            
        # Generate word importance for all dimensions using simpler method
        dimension_scores = {}
        for i, dimension in enumerate(DIMENSIONS):
            dimension_scores[dimension] = []
        
        # 1. Get the baseline prediction for the full text
        with torch.no_grad():
            inputs = tokenizer(
                review_text, 
                return_tensors="pt",
                truncation=True, 
                padding=True, 
                max_length=MAX_LENGTH
            )
            outputs = model(**inputs)
            predictions = torch.sigmoid(outputs.logits)
            baseline_scores = predictions.detach().numpy()[0]
        
        # 2. For each word, measure its importance by removing it
        for i, word in enumerate(words):
            if len(words) <= 1:  # Skip if only one word
                continue
                
            # Create text with this word removed
            words_without_i = words.copy()
            words_without_i.pop(i)
            modified_text = " ".join(words_without_i)
            
            # Get prediction without the word
            with torch.no_grad():
                mod_inputs = tokenizer(
                    modified_text, 
                    return_tensors="pt",
                    truncation=True, 
                    padding=True, 
                    max_length=MAX_LENGTH
                )
                mod_outputs = model(**mod_inputs)
                mod_predictions = torch.sigmoid(mod_outputs.logits)
                mod_scores = mod_predictions.detach().numpy()[0]
            
            # For each dimension, calculate importance as difference in scores
            for dim_idx, dimension in enumerate(DIMENSIONS):
                importance = float(baseline_scores[dim_idx] - mod_scores[dim_idx])
                dimension_scores[dimension].append({
                    "word": word,
                    "value": importance,
                    "is_positive": importance > 0
                })
        
        # 3. For each dimension, sort words by absolute importance and take top N
        top_words = {}
        for dimension in DIMENSIONS:
            if dimension_scores[dimension]:
                # Sort by absolute importance (largest effect first)
                sorted_words = sorted(
                    dimension_scores[dimension],
                    key=lambda x: abs(x["value"]),
                    reverse=True
                )
                # Take top N words
                top_words[dimension] = sorted_words[:request.top_n_words]
            else:
                top_words[dimension] = []
        
        # 4. Create visualization using matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from io import BytesIO
            import base64
            
            # Create visualization for the top dimension
            top_dim_idx = np.argmax(baseline_scores)
            top_dimension = DIMENSIONS[top_dim_idx]
            
            # Extract top words for visualization
            top_words_for_viz = top_words[top_dimension]
            
            # Create figure with a larger size for better visualization
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # Prepare data for visualization
            viz_words = [item["word"] for item in top_words_for_viz[:10]]
            viz_values = [item["value"] for item in top_words_for_viz[:10]]
            
            # Create horizontal bar chart with improved styling
            bars = ax.barh(
                viz_words,
                viz_values,
                color=['#FF4444' if v > 0 else '#3366CC' for v in viz_values],
                height=0.7,  # Slightly thinner bars for better spacing
                edgecolor='black',  # Add outline for better visibility
                linewidth=0.5,
                alpha=0.8
            )
            
            # Add labels and title with improved styling
            ax.set_title(f"Words influencing '{top_dimension}'", fontsize=16, fontweight='bold')
            ax.set_xlabel("Impact on classification score", fontsize=14)
            
            # Improve grid and background
            ax.set_axisbelow(True)  # Put gridlines below the bars
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.set_facecolor('#f9f9f9')  # Light background
            
            # Add a vertical line at x=0 with improved styling
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1.5)
            
            # Add value labels with improved styling
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.01 if width > 0 else width - 0.01
                ax.text(
                    label_x_pos, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}',
                    va='center',
                    ha='left' if width > 0 else 'right',
                    color='black',
                    fontweight='bold',
                    fontsize=10
                )
            
            # Add legend to explain colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#FF4444', edgecolor='black', label='Increases score'),
                Patch(facecolor='#3366CC', edgecolor='black', label='Decreases score')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            # Convert plot to HTML image with higher resolution
            buffer = BytesIO()
            fig.tight_layout()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', transparent=False)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            
            # Create HTML with both inline image and standalone image for flexibility
            html = f"""
            <div style="text-align: center;">
                <h3>Words influencing '{top_dimension}'</h3>
                <img src="data:image/png;base64,{img_str}" style="width:100%; max-width:1200px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);" />
                <p style="margin-top: 10px; font-style: italic;">Red bars indicate words that increase the prediction score, blue bars decrease it.</p>
            </div>
            """
            plt.close(fig)
            
        except Exception as viz_error:
            logger.error(f"Error creating visualization: {str(viz_error)}")
            html = f"<p>Could not generate visualization: {str(viz_error)}</p>"
        
        # Return the HTML and top words in the format expected by the frontend
        return {
            "explanation": {
                "html": html,
                "top_words": top_words
            }
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