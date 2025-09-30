#!/usr/bin/env python3
"""
Test script to verify chart display fixes in local deployment
"""
import requests
import json

# Define the URL for the local backend
API_URL = "http://localhost:8000"

def test_predict_api(review_text):
    """Test the predict API endpoint"""
    url = f"{API_URL}/predict"
    payload = {"review_text": review_text}
    
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("API response:")
        print(json.dumps(data, indent=2))
        
        # Format as percentages like in the frontend
        print("\nScores formatted as percentages (as displayed in frontend):")
        for item in data:
            label = item["label"]
            score = item["score"]
            score_pct = round(score * 100, 1)
            print(f"{label}: {score_pct}%")
        
        return data
    else:
        print(f"Error: {response.text}")
        return None

# Sample review
review = "We loved the eco-friendly resort that used solar power and served organic local food. Their conservation efforts were impressive!"

print("Testing the backend API...")
print(f"Review: {review}")
print("=" * 50)
test_predict_api(review)