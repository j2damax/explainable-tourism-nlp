# Serendip Experiential Engine

A containerized application for analyzing experiential dimensions in Sri Lankan tourism reviews, featuring:

- 🔍 **FastAPI Backend**: Handles text classification with NLP models and provides explainable results
- 🖥️ **Streamlit Frontend**: Interactive UI for submitting reviews and visualizing experiential dimensions
- 🐳 **Docker-Compose Setup**: Run both services concurrently with proper networking

## Architecture

```
serendip-experiential-engine/
├── backend/                # FastAPI application
│   ├── main.py             # Main entry point
│   ├── routers/            # API routes
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend container config
├── frontend/               # Streamlit application
│   ├── app.py              # Streamlit UI
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Frontend container config
└── docker-compose.yml      # Service orchestration
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Running the Application

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/explainable-tourism-nlp.git
   cd explainable-tourism-nlp/serendip-experiential-engine
   ```

2. Start the services using Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. Access the services:
   - Frontend (Streamlit): http://localhost:8501
   - Backend API (FastAPI): http://localhost:8000
   - Backend API docs: http://localhost:8000/docs

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

This application is designed to work with the tourism-review-classifier model. To integrate:

1. Import the trained model in the backend
2. Replace the mock prediction logic with actual model inference
3. Generate SHAP values for explanations

See the main project README for details on model training and evaluation.