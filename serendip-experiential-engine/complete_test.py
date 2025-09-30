#!/usr/bin/env python3
"""
Complete test script for frontend and backend integration
"""
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import sys
from datetime import datetime

# Configure URLs
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8501"

# Sample reviews for testing
SAMPLE_REVIEWS = [
    "We loved the eco-friendly resort that used solar power and served organic local food. Their conservation efforts were impressive!",
    "The yoga retreat by the beach offered amazing Ayurvedic treatments and meditation sessions that completely refreshed me.",
    "The cooking class taught us how to make authentic Sri Lankan curry and hoppers. We visited the spice market with the chef too!",
    "We hiked through remote villages in the mountains, staying with local families and seeing waterfalls that few tourists visit."
]

def check_health():
    """Check if the backend and frontend are running"""
    print("Checking backend health...")
    try:
        response = requests.get(BACKEND_URL, timeout=5)
        if response.status_code == 200:
            print(f"✅ Backend is running. Status: {response.json()}")
            return True
        else:
            print(f"❌ Backend returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend is not accessible: {e}")
        return False
    
def analyze_review(review_text):
    """Send a review to the backend for analysis"""
    print(f"\nAnalyzing review: {review_text[:50]}...")
    
    # Call predict endpoint
    try:
        predict_response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"review_text": review_text},
            timeout=30
        )
        
        print(f"Predict API response status: {predict_response.status_code}")
        
        if predict_response.status_code != 200:
            print(f"❌ Error: {predict_response.text}")
            return None
        
        # Format predictions
        predictions = predict_response.json()
        result = {
            "predictions": {item["label"]: item["score"] for item in predictions},
            "raw_response": predictions
        }
        
        # Print formatted results
        print("\nPrediction results:")
        for label, score in result["predictions"].items():
            print(f"  {label}: {score:.6f} ({round(score * 100, 1)}%)")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to backend: {e}")
        return None

def visualize_results(results, review_num):
    """Create a visualization of the results similar to the frontend"""
    if not results:
        return
    
    # Create a directory for the results if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    
    # Extract data
    labels = list(results["predictions"].keys())
    scores = list(results["predictions"].values())
    scores_pct = [round(score * 100, 1) for score in scores]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Dimension': labels,
        'Score (%)': scores_pct
    })
    
    # Sort by score
    df = df.sort_values('Score (%)', ascending=False)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    
    # Create horizontal bar chart
    ax = sns.barplot(x='Score (%)', y='Dimension', data=df, palette='viridis')
    
    # Add the values on the bars
    for i, v in enumerate(df['Score (%)']):
        ax.text(v + 0.5, i, f"{v}%", va='center')
    
    # Set labels and title
    plt.xlabel('Confidence Score (%)')
    plt.ylabel('')
    plt.title('Experience Dimension Analysis')
    
    # Fix x-axis range to 0-100
    plt.xlim(0, 100)
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results/analysis_{review_num}_{timestamp}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"\nVisualization saved as {filename}")
    
    # Create a table of key words by importance
    print("\nTop dimension: " + df['Dimension'].iloc[0])
    print(f"Score: {df['Score (%)'].iloc[0]}%")
    
    return filename

def main():
    """Main function to run the test"""
    print("=" * 50)
    print("Serendip Experiential Engine - Local Deployment Test")
    print("=" * 50)
    
    # Check if services are running
    if not check_health():
        print("❌ Please make sure the backend and frontend are running.")
        sys.exit(1)
    
    print("\n🔍 Testing with sample reviews...")
    
    for i, review in enumerate(SAMPLE_REVIEWS, 1):
        print("\n" + "=" * 50)
        print(f"REVIEW {i}/{len(SAMPLE_REVIEWS)}")
        print("=" * 50)
        
        # Analyze the review
        result = analyze_review(review)
        
        # Visualize the results
        if result:
            visualize_results(result, i)
            
        print("\n")
        time.sleep(1)  # Small delay between requests
    
    print("\n✅ All tests completed!")
    print(f"You can view the frontend at: {FRONTEND_URL}")
    print("Make sure the chart displays percentage values from 0-100% correctly")

if __name__ == "__main__":
    main()