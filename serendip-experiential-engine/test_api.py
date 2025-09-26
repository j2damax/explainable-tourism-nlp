#!/usr/bin/env python3
"""
API Test Script for Serendip Experiential Engine

This script provides a simple command-line utility to test
the Serendip API endpoints without requiring Postman.
"""

import argparse
import json
import requests
import sys
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

BASE_URL = "http://localhost:8000"

EXAMPLE_REVIEWS = {
    "eco": "We loved the sustainable eco-lodge in Ella. The property uses solar power, harvests rainwater, and serves organic food from their garden. The hosts educated us about local conservation efforts and we participated in a tree planting initiative during our stay.",
    
    "wellness": "The Ayurvedic spa retreat near Kandy was transformative. Daily yoga sessions at sunrise, meditation by the lake, and personalized herbal treatments helped me reconnect with myself. I've never felt more balanced and centered after a vacation.",
    
    "culinary": "Our cooking class in Galle was incredible! We visited the local market to select spices, learned traditional Sri Lankan curry recipes, and enjoyed our homemade hoppers and sambols. The chef explained the medicinal properties of each spice and herb used.",
    
    "adventure": "Hiking through Knuckles Mountain Range was challenging but rewarding. We crossed hanging bridges, discovered hidden waterfalls that weren't on any tourist map, and camped under the stars. Our local guide shared stories about the area's unique biodiversity.",
    
    "mixed": "Our trip to Sri Lanka combined adventure and wellness perfectly. We hiked through tea plantations in the morning, enjoyed farm-to-table Sri Lankan cuisine for lunch, and ended with Ayurvedic treatments. The eco-friendly resort used sustainable practices which we appreciated."
}

def parse_args():
    parser = argparse.ArgumentParser(description='Test Serendip API endpoints')
    parser.add_argument('--url', default=BASE_URL, help='Base URL for the API')
    parser.add_argument('--endpoint', choices=['predict', 'explain', 'health'], 
                        default='predict', help='Endpoint to test')
    parser.add_argument('--review_type', choices=list(EXAMPLE_REVIEWS.keys()), 
                       default='mixed', help='Type of example review to use')
    parser.add_argument('--custom_review', type=str, help='Custom review text to analyze')
    
    return parser.parse_args()

def test_health(base_url: str) -> None:
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        
        console.print(Panel.fit(
            f"[green]Health check successful![/green]\n"
            f"Status code: {response.status_code}\n"
            f"Response: {response.text}",
            title="Health Check"
        ))
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error testing health endpoint: {e}[/red]")
        sys.exit(1)

def test_predict(base_url: str, review: str) -> Dict[str, Any]:
    """Test the predict endpoint"""
    try:
        console.print(f"[yellow]Sending review to prediction API...[/yellow]")
        response = requests.post(
            f"{base_url}/predict",
            json={"review_text": review}
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Create table for results
        table = Table(title="Classification Results")
        table.add_column("Label", style="cyan")
        table.add_column("Score", style="magenta")
        
        # Sort results by score (descending)
        sorted_results = sorted(result, key=lambda x: x['score'], reverse=True)
        
        for item in sorted_results:
            score_percentage = f"{item['score']*100:.2f}%"
            table.add_row(item['label'], score_percentage)
            
        console.print(table)
        return result
        
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error testing predict endpoint: {e}[/red]")
        if hasattr(e.response, 'text'):
            console.print(f"Response: {e.response.text}")
        sys.exit(1)

def test_explain(base_url: str, review: str) -> None:
    """Test the explain endpoint"""
    try:
        console.print(f"[yellow]Getting explanations for review...[/yellow]")
        response = requests.post(
            f"{base_url}/explain",
            json={"review_text": review}
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Since we can't display HTML in the terminal
        console.print(Panel.fit(
            f"[green]Explanation successfully retrieved![/green]\n"
            f"SHAP HTML explanation length: {len(result.get('explanation', ''))} characters\n"
            f"To view the visualization, please use the Streamlit frontend",
            title="Explanation Results"
        ))
            
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error testing explain endpoint: {e}[/red]")
        if hasattr(e.response, 'text'):
            console.print(f"Response: {e.response.text}")
        sys.exit(1)

def main():
    args = parse_args()
    
    console.print(Panel.fit(
        f"[bold blue]Serendip Experiential Engine API Tester[/bold blue]\n"
        f"Base URL: {args.url}\n"
        f"Testing endpoint: {args.endpoint}",
        title="Test Configuration"
    ))
    
    # Determine which review to use
    if args.custom_review:
        review = args.custom_review
        console.print(f"Using custom review: {review[:50]}...")
    else:
        review = EXAMPLE_REVIEWS[args.review_type]
        console.print(f"Using example {args.review_type} review: {review[:50]}...")
    
    # Test appropriate endpoint
    if args.endpoint == 'health':
        test_health(args.url)
    elif args.endpoint == 'predict':
        test_predict(args.url, review)
    elif args.endpoint == 'explain':
        test_explain(args.url, review)

if __name__ == "__main__":
    main()