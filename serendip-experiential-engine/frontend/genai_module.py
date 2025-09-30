"""
GenAI Module for Benchmarking Classification
This module provides a few-shot learning based classification approach using the OpenAI API
"""
import os
import json
import openai
from typing import Dict, List, Any

# Import configuration
from config import DIMENSIONS, SCORE_THRESHOLDS, SAMPLE_REVIEWS, OPENAI_MODEL_DEFAULT, COST_PER_1K_TOKENS

# Initialize OpenAI client with API key validation
from dotenv import load_dotenv
from api_key_fix import get_openai_api_key

# Try to load from .env file first (for local development)
load_dotenv()

# Try multiple approaches to get the API key
api_key = get_openai_api_key()

# Debug information
print(f"DEBUG: OpenAI API key found: {'Yes' if api_key else 'No'}")
print(f"DEBUG: OpenAI API key length: {len(api_key) if api_key else 0}")
if api_key:
    print(f"DEBUG: First few characters of key: {api_key[:5]}... (if available)")

if not api_key:
    print("WARNING: OpenAI API key not found in environment variables.")
    print("GenAI comparison functionality will not be available.")

client = openai.OpenAI(api_key=api_key)

# Define example reviews and their classifications based on the samples from config
EXAMPLE_REVIEWS = [
    {
        "review": SAMPLE_REVIEWS[0],  # Eco-friendly resort
        "dimensions": {
            DIMENSIONS[0]["name"]: "HIGH",  # Regenerative & Eco-Tourism
            DIMENSIONS[1]["name"]: "LOW",   # Integrated Wellness
            DIMENSIONS[2]["name"]: "MEDIUM", # Immersive Culinary
            DIMENSIONS[3]["name"]: "LOW"    # Off-the-Beaten-Path Adventure
        }
    },
    {
        "review": SAMPLE_REVIEWS[1],  # Yoga retreat
        "dimensions": {
            DIMENSIONS[0]["name"]: "LOW",   # Regenerative & Eco-Tourism
            DIMENSIONS[1]["name"]: "HIGH",  # Integrated Wellness
            DIMENSIONS[2]["name"]: "LOW",   # Immersive Culinary
            DIMENSIONS[3]["name"]: "LOW"    # Off-the-Beaten-Path Adventure
        }
    },
    {
        "review": SAMPLE_REVIEWS[3],  # Remote villages hiking
        "dimensions": {
            DIMENSIONS[0]["name"]: "MEDIUM", # Regenerative & Eco-Tourism
            DIMENSIONS[1]["name"]: "LOW",    # Integrated Wellness
            DIMENSIONS[2]["name"]: "LOW",    # Immersive Culinary
            DIMENSIONS[3]["name"]: "HIGH"    # Off-the-Beaten-Path Adventure
        }
    }
]

def run_genai_benchmark(review_text: str) -> Dict[str, Any]:
    """
    Perform a few-shot classification using OpenAI API
    
    Args:
        review_text: The tourism review text to classify
        
    Returns:
        Dictionary with classifications and metadata
    """
    # Check if API key is available
    if not api_key:
        result = {dim["name"]: "API_KEY_MISSING" for dim in DIMENSIONS}
        result["error"] = "OpenAI API key not set in environment variables"
        result["metadata"] = {"model": "none", "tokens_used": 0, "estimated_cost": 0.0}
        return result
    
    # Construct the prompt with examples (few-shots)
    system_prompt = "You are an AI trained to classify tourism reviews based on four experiential dimensions:\n"
    
    # Add dimensions from config
    for i, dim in enumerate(DIMENSIONS, 1):
        system_prompt += f"{i}. {dim['name']}: {dim['description']}\n"
    
    system_prompt += "\nFor each dimension, classify the review as:\n"
    system_prompt += "- HIGH: Strong presence of this dimension\n"
    system_prompt += "- MEDIUM: Some presence of this dimension\n"
    system_prompt += "- LOW: Little to no presence of this dimension\n\n"
    system_prompt += "Return only the JSON output with your classifications."
    
    # Create examples text
    examples_text = "Here are some examples of classified reviews:\n\n"
    for example in EXAMPLE_REVIEWS:
        examples_text += f"REVIEW: {example['review']}\n"
        examples_text += f"CLASSIFICATION: {json.dumps(example['dimensions'], indent=2)}\n\n"
    
    user_prompt = f"{examples_text}\nREVIEW: {review_text}\n\nCLASSIFICATION:"
    
    try:
        # Get model from environment or use default from config
        model = os.environ.get("OPENAI_MODEL", OPENAI_MODEL_DEFAULT)
        
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Lower temperature for more consistent results
            max_tokens=200
        )
        
        # Extract JSON from response
        raw_result = response.choices[0].message.content.strip()
        
        # Parse the JSON output
        try:
            # Try to parse directly
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from text
            json_start = raw_result.find('{')
            json_end = raw_result.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw_result[json_start:json_end]
                result = json.loads(json_str)
            else:
                # If extraction fails, create a structured response with dimension names from config
                result = {dim["name"]: "UNKNOWN" for dim in DIMENSIONS}
                result["error"] = "Failed to parse classification"
        
        # Add metadata
        # Use cost per token from config
        cost_per_1k = COST_PER_1K_TOKENS.get(model, COST_PER_1K_TOKENS[OPENAI_MODEL_DEFAULT])
            
        result["metadata"] = {
            "model": model,
            "tokens_used": response.usage.total_tokens,
            "estimated_cost": round(response.usage.total_tokens / 1000 * cost_per_1k, 4)
        }
        
        return result
    
    except Exception as e:
        # Create error response using dimension names from config
        error_response = {dim["name"]: "ERROR" for dim in DIMENSIONS}
        error_response["error"] = str(e)
        return error_response


def compare_results(bert_results: Dict[str, float], genai_results: Dict[str, str]) -> Dict[str, Any]:
    """Compare BERT and GenAI classification results."""
    # Map GenAI categorical values to numeric for comparison using config
    category_to_numeric = {
        "HIGH": SCORE_THRESHOLDS["high"],
        "MEDIUM": SCORE_THRESHOLDS["medium"],
        "LOW": SCORE_THRESHOLDS["low"],
        "ERROR": 0.0,
        "UNKNOWN": 0.0
    }
    
    # Extract dimensions (excluding metadata/error keys)
    dimensions = [dim for dim in genai_results.keys() 
                 if dim not in ["metadata", "error"]]
    
    # Calculate agreement
    agreement_count = 0
    dimension_comparisons = {}
    
    for dim in dimensions:
        bert_score = bert_results.get(dim, 0.0)
        genai_category = genai_results.get(dim, "UNKNOWN")
        genai_score = category_to_numeric.get(genai_category, 0.0)
        
        # Map BERT score to category for comparison using thresholds from config
        bert_category = "LOW"
        if bert_score > SCORE_THRESHOLDS["high"]:
            bert_category = "HIGH"
        elif bert_score > SCORE_THRESHOLDS["medium"]:
            bert_category = "MEDIUM"
        
        # Check if categories agree
        agrees = bert_category == genai_category
        if agrees:
            agreement_count += 1
        
        dimension_comparisons[dim] = {
            "bert_score": bert_score,
            "bert_category": bert_category,
            "genai_category": genai_category,
            "agrees": agrees
        }
    
    agreement_percentage = (agreement_count / len(dimensions)) * 100 if dimensions else 0
    
    # Get metadata from GenAI results
    metadata = genai_results.get("metadata", {})
    tokens_used = metadata.get("tokens_used", 0)
    estimated_cost = metadata.get("estimated_cost", 0)
    
    # Comparison metrics
    comparison = {
        "agreement_percentage": agreement_percentage,
        "dimensions": dimension_comparisons,
        "cost_comparison": {
            "bert": {
                "inference_time": "~100ms",
                "cost_per_1k_reviews": "$0.01",  # Estimated hosting costs
                "reproducibility": "High (deterministic)",
                "explanation": "Local model, fixed costs regardless of usage"
            },
            "genai": {
                "inference_time": "~1-2s",
                "tokens_used": tokens_used,
                "cost_per_review": f"${estimated_cost}",
                "cost_per_1k_reviews": f"${round(estimated_cost * 1000, 2)}",
                "reproducibility": "Medium (temperature affects outputs)",
                "explanation": "API costs scale with usage"
            }
        }
    }
    
    return comparison