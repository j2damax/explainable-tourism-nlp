"""
API Service module for Serendip Experiential Engine Frontend
Centralizes all API calls and error handling
"""
import requests
import streamlit as st
import time
import json
from typing import Dict, Tuple, List, Any, Optional

# Import configuration
from config import API_URL, API_TIMEOUT_SHORT, API_TIMEOUT_LONG

def api_call_with_retry(func, *args, max_retries=3, **kwargs):
    """
    Execute an API call with retry capability
    
    Args:
        func: The function to call
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retries
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call
    """
    retries = 0
    last_exception = None
    
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            last_exception = e
            retries += 1
            if retries >= max_retries:
                break
            st.warning(f"Retrying API call ({retries}/{max_retries})...")
            time.sleep(1)  # Wait before retry
    
    # If we get here, all retries failed
    raise last_exception

def check_health() -> Dict[str, Any]:
    """
    Check if the backend API is available and model is loaded
    
    Returns:
        Dictionary with status information
    """
    try:
        response = requests.get(f"{API_URL}/", timeout=API_TIMEOUT_SHORT)
        print(f"Health check response: {response.status_code}")
        print(f"Health check response body: {response.text}")
        
        if response.status_code != 200:
            return {
                "available": False, 
                "model_loaded": False,
                "message": f"Backend API is not available. Status: {response.status_code}"
            }
        
        model_status = response.json().get("model_status", "unknown")
        if model_status != "loaded":
            return {
                "available": True, 
                "model_loaded": False,
                "message": "Model is still loading. Your first request may take a minute to process."
            }
        
        return {
            "available": True, 
            "model_loaded": True,
            "message": "API is ready"
        }
    except requests.exceptions.RequestException as e:
        return {
            "available": False, 
            "model_loaded": False,
            "message": f"Cannot connect to backend API: {str(e)}"
        }

def get_dimensions() -> Dict[str, Any]:
    """
    Fetch available dimensions from the API
    
    Returns:
        Dictionary with success status and dimensions list
    """
    try:
        response = api_call_with_retry(
            requests.get,
            f"{API_URL}/dimensions",
            timeout=API_TIMEOUT_SHORT
        )
        
        print(f"Dimensions response: {response.status_code}")
        #print(f"Dimensions response body: {response.text}")
        
        if response.status_code == 200:
            return {
                "success": True,
                "dimensions": response.json().get("dimensions", [])
            }
        else:
            return {
                "success": False,
                "dimensions": []
            }
    except Exception as e:
        return {
            "success": False,
            "dimensions": []
        }

def analyze_review(review_text: str) -> Optional[Dict[str, Any]]:
    """
    Send review to API for analysis and explanation
    
    Args:
        review_text: The review text to analyze
        
    Returns:
        Dictionary with predictions and explanation or None if an error occurred
    """
    try:
        # Check API health first
        api_status = check_health()
        if not api_status["available"]:
            st.error(api_status["message"])
            return None
        elif not api_status["model_loaded"]:
            st.warning(api_status["message"])
        
        # Call predict endpoint
        predict_response = api_call_with_retry(
            requests.post,
            f"{API_URL}/predict",
            json={"review_text": review_text},
            timeout=API_TIMEOUT_LONG
        )
        
        print(f"Predict response: {predict_response.status_code}")
        print(f"Predict response body: {predict_response.text[:500]}...")
        
        if predict_response.status_code != 200:
            st.error(f"Error analyzing review: {predict_response.status_code}")
            try:
                error_detail = predict_response.json().get("detail", "No details provided")
                st.error(f"Error details: {error_detail}")
            except:
                pass
            return None
        
        # Call explain endpoint with retry mechanism
        explain_response = api_call_with_retry(
            requests.post,
            f"{API_URL}/explain",
            json={"review_text": review_text, "top_n_words": 5},
            timeout=API_TIMEOUT_SHORT
        )
        
        print(f"Explain response: {explain_response.status_code}")
        print(f"Explain response body length: {len(explain_response.text)} characters")
        print(f"Explain response content type: {explain_response.headers.get('Content-Type', 'unknown')}")
        
        # Format response
        predictions = predict_response.json()
        print(f"Formatted predictions: {json.dumps(predictions, indent=2)[:500]}...")
        result = {
            "predictions": {item["label"]: item["score"] for item in predictions},
            "explanation": {}
        }
        
        if explain_response.status_code == 200:
            explanation_data = explain_response.json().get("explanation", {})
            result["explanation"] = explanation_data
            
            # Extract the base64 image data from HTML if present
            html_content = explanation_data.get("html", "")
            if "base64" in html_content:
                try:
                    # Extract base64 image data
                    import re
                    base64_pattern = re.compile(r'src="data:image/png;base64,([^"]+)"')
                    matches = base64_pattern.search(html_content)
                    if matches:
                        base64_img = matches.group(1)
                        result["explanation"]["base64_img"] = base64_img
                        print("Successfully extracted base64 image data as fallback")
                except Exception as e:
                    print(f"Error extracting base64 image: {str(e)}")
        else:
            st.warning(f"Could not generate explanations: {explain_response.status_code}")
        
        return result
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        st.info("If this is your first analysis, the model may still be loading. Please try again in a minute.")
        return None