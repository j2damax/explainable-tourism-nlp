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

Interactive Streamlit web application for analyzing tourism reviews across experiential dimensions, providing visualizations and explanations of model predictions.

## Architecture

This frontend is built using [Streamlit](https://streamlit.io/) and serves as the user interface for the Serendip Experiential Engine. It communicates with the backend API to get predictions and explanations for tourism reviews.

## Features

- **Interactive Review Analysis**: Input or select sample reviews for instant analysis
- **Multi-dimensional Visualization**: Radar charts showing prediction scores across all dimensions
- **Explainability**: SHAP-based word-level explanations showing which words influence each dimension
- **GenAI Benchmark**: Compare transformer model predictions with OpenAI GPT responses
- **Sample Gallery**: Pre-loaded examples showcasing different experiential patterns

## Experiential Dimensions

- 🌱 **Regenerative & Eco-Tourism**: Sustainable travel with positive environmental and social impact
- 🧘 **Integrated Wellness**: Experiences focused on physical and mental well-being
- 🍜 **Immersive Culinary**: Authentic local cuisine and food-related experiences
- 🌄 **Off-the-Beaten-Path Adventure**: Exploration of less-crowded natural areas and unique activities

## Project Structure

```
frontend/
├── app.py                # Main Streamlit application
├── components/           # UI components
│   ├── header.py         # Page header and navigation
│   ├── results.py        # Results visualization 
│   └── shap_viz.py       # SHAP explanations visualization
├── utils/                # Utility functions
│   ├── api.py            # Backend API communication
│   ├── openai_client.py  # OpenAI API integration
│   └── sample_data.py    # Sample reviews for demo
└── assets/               # Static assets (images, CSS)
```

## Setup and Configuration

### Environment Variables

Create a `.env` file with the following variables:

```
API_URL=http://backend:8000
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

### Deployment on Hugging Face Spaces

1. Fork the repository
2. Create a new Space on Hugging Face Spaces
3. Link to your repository
4. Set the required environment variables in the Space settings

## API Integration

The frontend communicates with the backend API to:
1. Submit reviews for prediction
2. Get prediction scores for each dimension
3. Retrieve SHAP explanations for model decisions

## External Services

- **Backend API**: Provides model predictions and explanations
- **OpenAI API**: Used for GenAI comparison feature (optional)

## Links

- Backend API: [Serendip Experiential Backend](https://huggingface.co/spaces/j2damax/serendip-experiential-backend)
- Live Demo: [Serendip Experiential Frontend](https://huggingface.co/spaces/j2damax/serendip-experiential-frontend)
- GitHub Repository: [explainable-tourism-nlp](https://github.com/j2damax/explainable-tourism-nlp)
