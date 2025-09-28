"""
Configuration module for Serendip Experiential Engine Frontend
Contains all constants and settings used across the application
"""
import os

# API Configuration
# For local development: "http://backend:8000" or "http://localhost:8000"
# For Hugging Face deployment: use the URL of your backend Space
API_URL = os.environ.get("API_URL", "https://j2damax-serendip-experiential-backend.hf.space")
API_TIMEOUT_SHORT = int(os.environ.get("API_TIMEOUT_SHORT", 30))
API_TIMEOUT_LONG = int(os.environ.get("API_TIMEOUT_LONG", 60))

# Dimension Definitions
DIMENSIONS = [
    {"id": "regen_eco", "name": "Regenerative & Eco-Tourism", "description": "Travel focused on positive social/environmental impact", "icon": "🌱"},
    {"id": "wellness", "name": "Integrated Wellness", "description": "Journeys combining physical and mental well-being", "icon": "🧘"},
    {"id": "culinary", "name": "Immersive Culinary", "description": "Experiences centered on authentic local cuisine", "icon": "🍜"},
    {"id": "adventure", "name": "Off-the-Beaten-Path Adventure", "description": "Exploration of less-crowded natural landscapes", "icon": "🌄"}
]

# Visualization Settings
SCORE_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.3,
    "low": 0
}

# Sample Data
SAMPLE_REVIEWS = [
    "We loved the eco-friendly resort that used solar power and served organic local food. Their conservation efforts were impressive!",
    "The yoga retreat by the beach offered amazing Ayurvedic treatments and meditation sessions that completely refreshed me.",
    "The cooking class taught us how to make authentic Sri Lankan curry and hoppers. We visited the spice market with the chef too!",
    "We hiked through remote villages in the mountains, staying with local families and seeing waterfalls that few tourists visit."
]

# UI Configuration
APP_TITLE = "Serendip Experiential Engine"
APP_ICON = "✨"
LOGO_URL = "https://placehold.co/200x100/1E88E5/FFFFFF?text=SERENDIP"

# Structured Configurations
UI_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "logo_url": LOGO_URL,
    "genai_enabled": os.environ.get("OPENAI_API_KEY") is not None
}

API_CONFIG = {
    "url": API_URL,
    "timeout_short": API_TIMEOUT_SHORT,
    "timeout_long": API_TIMEOUT_LONG
}

# GenAI Configuration
OPENAI_MODEL_DEFAULT = "gpt-3.5-turbo"
COST_PER_1K_TOKENS = {
    "gpt-3.5-turbo": 0.001,
    "gpt-4": 0.06,
    "gpt-4-turbo": 0.01
}