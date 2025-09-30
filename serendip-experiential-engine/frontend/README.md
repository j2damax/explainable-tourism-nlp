---
title: Serendip Experiential Frontend
emoji: ✨
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# Serendip Experiential Frontend

This Space hosts the Streamlit frontend for the Serendip Experiential Engine, providing an interactive UI for analyzing experiential dimensions in Sri Lankan tourism reviews.

## Environment Variables

The frontend requires the following environment variables:

- `API_URL`: URL of the backend API (default: http://backend:8000)
- `OPENAI_API_KEY`: API key for OpenAI services (required for GenAI Classification feature)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-3.5-turbo)

### Troubleshooting "API_KEY_MISSING" Error

If the GenAI Classification feature shows "API_KEY_MISSING", ensure:

1. The OPENAI_API_KEY is properly set in the `.env` file in the project root
2. The environment variable is correctly passed to the container
3. The container has access to the environment variable

## Features

- Submit tourism reviews for analysis
- Visualize experiential dimension scores
- View explainable AI insights using SHAP
- Compare with GenAI benchmark responses

## Experiential Dimensions

The application analyzes tourism reviews across four key experiential dimensions:

- 🌱 **Regenerative & Eco-Tourism**: Travel focused on positive social/environmental impact
- 🧘 **Integrated Wellness**: Journeys combining physical and mental well-being
- 🍜 **Immersive Culinary**: Experiences centered on authentic local cuisine
- 🌄 **Off-the-Beaten-Path Adventure**: Exploration of less-crowded natural landscapes

## Backend API

This frontend connects to the [Serendip Experiential Backend](https://huggingface.co/spaces/j2damax/serendip-experiential-backend), which hosts the classification model and explainability features.

## Technologies

- Streamlit
- SHAP for explainability visualization
- Plotly for interactive charts

---

<a href="https://github.com/j2damax/explainable-tourism-nlp" target="_blank">View on GitHub</a>
