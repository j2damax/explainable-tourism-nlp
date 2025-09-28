# Serendip Experiential Engine

A containerized application for analyzing experiential dimensions in Sri Lankan tourism reviews, featuring:

- 🔍 **FastAPI Backend**: Handles text classification with NLP models and provides explainable results
- 🖥️ **Streamlit Frontend**: Interactive UI for submitting reviews and visualizing experiential dimensions
- 🐳 **Docker-Compose Setup**: Run both services concurrently with proper networking

## Architecture

```
serendip-experiential-engine/
├── backend/                # FastAPI application
│   ├── api.py              # Main API implementation
│   ├── main.py             # Entry point
│   └── requirements.txt    # Backend dependencies
├── frontend/               # Streamlit application
│   ├── app.py              # Streamlit UI
│   ├── genai_module.py     # GenAI benchmark functionality
│   └── requirements.txt    # Frontend dependencies
├── notebooks/              # Jupyter notebooks
│   └── api_usage_example.ipynb # Example of programmatic API usage
├── Makefile                # Helpful commands for development
└── docker-compose.yml      # Service orchestration
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Quick Start

1. Clone this repository
2. Navigate to the project directory
3. Configure the environment variables in `.env` file (especially the OpenAI API key for GenAI features)
4. Run `docker compose up` or `make up`
5. Open your browser to `http://localhost:8501`

### Development Commands

Using the included Makefile:

- `make build`: Build the Docker containers
- `make up`: Start the services
- `make down`: Stop the services
- `make test`: Run the tests
- `make backend-dev`: Run the backend service locally
- `make frontend-dev`: Run the frontend service locally

### Development

#### Backend Development

The backend is built with FastAPI and provides endpoints for:

- Classifying tourism reviews into experiential dimensions
- Providing explainable AI insights on why a review was classified a certain way

To extend the backend:

1. Add new routes in the `routers/` directory
2. Update `main.py` to include new routers
3. Install any new dependencies in `requirements.txt`

#### Frontend Development

The Streamlit frontend provides:

- A user interface for submitting reviews for analysis
- Visualization of classification results and explanations
- Sample reviews to demonstrate the system

To extend the frontend:

1. Modify `app.py` to add new UI components or visualizations
2. Update API calls to backend as needed
3. Install any new dependencies in `requirements.txt`

## Environment Variables

Backend:

- `PORT`: API port (default: 8000)
- `HOST`: API host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: info)

Frontend:

- `API_URL`: URL of the backend API (default: http://backend:8000)

## Integrating with the ML Model

This application is designed to work with the Hugging Face model j2damax/serendip-travel-classifier. To integrate:

1. Import the trained model in the backend from Hugging Face Hub
2. Set up the prediction pipeline for multi-label classification
3. Generate explanations using the word impact analysis

See the main project README for details on model training and evaluation.
