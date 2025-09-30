#!/usr/bin/env python3
"""
Test script to verify data flow between backend and frontend
This script makes direct calls to the backend API and compares the data with frontend rendering
"""

import requests
import json
import pandas as pd
import streamlit as st
import os
import sys
from typing import Dict, Any

# Add the frontend and backend directories to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import frontend config and API service
from frontend.config import API_URL, DIMENSIONS
from frontend.api_service import analyze_review

def test_direct_api_call(review_text: str) -> Dict[str, Any]:
    """Make a direct call to the backend API"""
    print(f"Making direct API call to {API_URL}/predict")
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"review_text": review_text},
            timeout=30
        )
        
        print(f"API Response status: {response.status_code}")
        print(f"API Response content: {response.text}")
        
        if response.status_code == 200:
            predictions = response.json()
            return {
                "predictions": {item["label"]: item["score"] for item in predictions},
                "raw_response": predictions
            }
        else:
            print(f"Error: API returned status code {response.status_code}")
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        print(f"Exception during API call: {str(e)}")
        return {"error": f"Exception: {str(e)}"}

def compare_api_frontend(review_text: str) -> Dict[str, Any]:
    """Compare direct API results with frontend processing"""
    # Get direct API results
    direct_results = test_direct_api_call(review_text)
    
    # Get frontend processed results
    frontend_results = analyze_review(review_text)
    
    if "error" in direct_results or not frontend_results:
        return {
            "success": False,
            "direct_api": direct_results,
            "frontend": frontend_results
        }
    
    # Compare the results
    comparison = {}
    for dimension in DIMENSIONS:
        dim_name = dimension["name"]
        if dim_name in direct_results["predictions"] and dim_name in frontend_results["predictions"]:
            direct_value = direct_results["predictions"][dim_name]
            frontend_value = frontend_results["predictions"][dim_name]
            
            # Calculate how the frontend would display this value
            display_value_pct = round(frontend_value * 100, 1)
            
            comparison[dim_name] = {
                "direct_api_value": direct_value,
                "frontend_value": frontend_value,
                "frontend_display_value": f"{display_value_pct}%",
                "match": abs(direct_value - frontend_value) < 0.0001  # Check for floating point equality
            }
    
    return {
        "success": True,
        "comparison": comparison,
        "direct_api": direct_results,
        "frontend": frontend_results
    }

def main():
    """Main function to run the test"""
    print("===== Serendip Experiential Engine - Backend to Frontend Data Flow Test =====")
    
    # Test with a sample review
    sample_review = "We loved the eco-friendly resort that used solar power and served organic local food. Their conservation efforts were impressive!"
    
    print(f"\nTesting with sample review: \"{sample_review}\"\n")
    
    results = compare_api_frontend(sample_review)
    
    if not results["success"]:
        print("❌ Test failed - could not get results from API or frontend")
        if "error" in results["direct_api"]:
            print(f"   API error: {results['direct_api']['error']}")
        return
    
    print("\n===== Results =====\n")
    
    # Create a comparison table
    comparison_data = []
    all_match = True
    
    for dimension, data in results["comparison"].items():
        comparison_data.append({
            "Dimension": dimension,
            "API Value": f"{data['direct_api_value']:.6f}",
            "Frontend Value": f"{data['frontend_value']:.6f}",
            "Display Value": data['frontend_display_value'],
            "Match": "✅" if data["match"] else "❌"
        })
        if not data["match"]:
            all_match = False
    
    # Print as a nicely formatted table
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    print("\nTest Summary:")
    if all_match:
        print("✅ All values match between API and frontend")
    else:
        print("❌ Some values don't match between API and frontend")
    
    print("\n===== Raw API Response =====\n")
    print(json.dumps(results["direct_api"]["raw_response"], indent=2))
    
    print("\n===== Frontend Processed Data =====\n")
    print(json.dumps(results["frontend"]["predictions"], indent=2))

if __name__ == "__main__":
    main()