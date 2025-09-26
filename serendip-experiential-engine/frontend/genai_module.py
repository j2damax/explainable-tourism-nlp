"""
GenAI Module for Benchmarking Classification
This module provides a few-shot learning based classification approach using the OpenAI API
"""
import os
import json
import openai
from typing import Dict, List, Any

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# Define example reviews and their classifications
EXAMPLE_REVIEWS = [
    {
        "review": "We loved the eco-friendly resort that used solar power and served organic local food. Their conservation efforts were impressive!",
        "dimensions": {
            "Regenerative & Eco-Tourism": "HIGH", 
            "Integrated Wellness": "LOW", 
            "Immersive Culinary": "MEDIUM",
            "Off-the-Beaten-Path Adventure": "LOW"
        }
    },
    {
        "review": "The yoga retreat by the beach offered amazing Ayurvedic treatments and meditation sessions that completely refreshed me.",
        "dimensions": {
            "Regenerative & Eco-Tourism": "LOW", 
            "Integrated Wellness": "HIGH", 
            "Immersive Culinary": "LOW",
            "Off-the-Beaten-Path Adventure": "LOW"
        }
    },
    {
        "review": "We hiked through remote villages in the mountains, staying with local families and seeing waterfalls that few tourists visit.",
        "dimensions": {
            "Regenerative & Eco-Tourism": "MEDIUM", 
            "Integrated Wellness": "LOW", 
            "Immersive Culinary": "LOW",
            "Off-the-Beaten-Path Adventure": "HIGH"
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
    # Construct the prompt with examples (few-shots)
    system_prompt = """
    You are an AI trained to classify tourism reviews based on four experiential dimensions:
    1. Regenerative & Eco-Tourism: Travel focused on positive social/environmental impact
    2. Integrated Wellness: Journeys combining physical and mental well-being
    3. Immersive Culinary: Experiences centered on authentic local cuisine
    4. Off-the-Beaten-Path Adventure: Exploration of less-crowded natural landscapes
    
    For each dimension, classify the review as:
    - HIGH: Strong presence of this dimension
    - MEDIUM: Some presence of this dimension
    - LOW: Little to no presence of this dimension
    
    Return only the JSON output with your classifications.
    """
    
    # Create examples text
    examples_text = "Here are some examples of classified reviews:\n\n"
    for example in EXAMPLE_REVIEWS:
        examples_text += f"REVIEW: {example['review']}\n"
        examples_text += f"CLASSIFICATION: {json.dumps(example['dimensions'], indent=2)}\n\n"
    
    user_prompt = f"{examples_text}\nREVIEW: {review_text}\n\nCLASSIFICATION:"
    
    try:
        # Get model from environment or use gpt-3.5-turbo as default (more affordable)
        model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Make API call
        response = client.chat.completions.create(
            model=model,  # Use model from environment variable
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
                # If extraction fails, create a structured response
                result = {
                    "Regenerative & Eco-Tourism": "UNKNOWN",
                    "Integrated Wellness": "UNKNOWN",
                    "Immersive Culinary": "UNKNOWN",
                    "Off-the-Beaten-Path Adventure": "UNKNOWN",
                    "error": "Failed to parse classification"
                }
        
        # Add metadata
        model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        # Calculate cost based on model
        cost_per_1k = 0.001  # Default rate for gpt-3.5-turbo
        if "gpt-4" in model:
            cost_per_1k = 0.06  # Rate for gpt-4
            
        result["metadata"] = {
            "model": model,
            "tokens_used": response.usage.total_tokens,
            "estimated_cost": round(response.usage.total_tokens / 1000 * cost_per_1k, 4)
        }
        
        return result
    
    except Exception as e:
        return {
            "Regenerative & Eco-Tourism": "ERROR",
            "Integrated Wellness": "ERROR",
            "Immersive Culinary": "ERROR",
            "Off-the-Beaten-Path Adventure": "ERROR",
            "error": str(e)
        }


def compare_results(bert_results: Dict[str, float], genai_results: Dict[str, str]) -> Dict[str, Any]:
    """
    Compare BERT and GenAI classification results
    
    Args:
        bert_results: Dictionary with BERT scores (0-1)
        genai_results: Dictionary with GenAI classifications (HIGH, MEDIUM, LOW)
        
    Returns:
        Dictionary with comparison metrics
    """
    # Map GenAI categorical values to numeric for comparison
    category_to_numeric = {
        "HIGH": 0.85,
        "MEDIUM": 0.5,
        "LOW": 0.15,
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
        
        # Map BERT score to category for comparison
        bert_category = "LOW"
        if bert_score > 0.7:
            bert_category = "HIGH"
        elif bert_score > 0.3:
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