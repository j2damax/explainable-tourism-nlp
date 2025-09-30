# Serendip Experiential Engine

A containerized web application for analyzing experiential dimensions in tourism reviews.

## Components

- 🔍 **FastAPI Backend**: Serves the NLP model with SHAP explainability
- 🖥️ **Streamlit Frontend**: Interactive UI for visualization
- 🐳 **Docker-Compose Setup**: Easy deployment of both services

## Features

- Tourism review analysis and classification
- Experience dimension visualization
- Word-level explanation using SHAP
- Benchmark comparison with GenAI responses

## Quick Start

1. Set up environment:
   ```bash
   cp .env.example .env
   # Edit .env and add OPENAI_API_KEY
   ```

2. Run with Docker:
   ```bash
   docker compose up
   ```

3. Open in browser:
   ```
   http://localhost:8501
   ```

## Deployment

Deploy to Hugging Face Spaces:

```bash
export HF_TOKEN=your_huggingface_token
./deploy_all.sh
```

## Live Demo

- [Frontend](https://huggingface.co/spaces/j2damax/serendip-experiential-frontend)
- [Backend](https://huggingface.co/spaces/j2damax/serendip-experiential-backend)

## Key Commands

- `make build`: Build containers
- `make up`: Start services
- `make down`: Stop services

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

## Hugging Face Spaces Deployment

This application can be deployed to Hugging Face Spaces. For detailed instructions, see [HUGGINGFACE_DEPLOYMENT.md](./HUGGINGFACE_DEPLOYMENT.md).

### Managing Storage Limits

When deployed on Hugging Face Spaces, you may encounter the "Workload evicted, storage limit exceeded (50G)" error. To address this:

1. **Use the provided reset script**:

   ```bash
   ./deploy_reset_cache.sh
   ```

   This script cleans temporary files, rebuilds the containers, and restarts the application with optimized settings.

2. **Monitor storage usage**:

   ```bash
   ./monitor_storage.sh
   ```

   This interactive tool shows current disk usage, largest directories, and memory consumption.

3. **Storage optimization features**:
   - Temporary model storage in `/tmp` directory
   - Automatic cache cleanup
   - Reduced image quality for visualizations
   - Memory-optimized model loading
   - No persistent volumes in production

If you encounter persistent issues, consult the troubleshooting section in [HUGGINGFACE_DEPLOYMENT.md](./HUGGINGFACE_DEPLOYMENT.md).
