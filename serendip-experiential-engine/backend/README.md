# Serendip Experiential Backend

FastAPI service that provides tourism review classification and explainability.

## Features

- BERT-based multi-label classification
- SHAP explainability for predictions
- RESTful API for analysis requests

## API Endpoints

- `POST /analyze`: Analyze a tourism review text
- `GET /health`: Check service health
- `GET /info`: Get model information

## Environment Variables

- `PORT`: Server port (default: 8000)
- `HOST`: Host to bind (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: info)

## Deployment

```bash
export HF_TOKEN=your_huggingface_token
./deploy_to_huggingface.sh
```

## Live Demo

API is deployed at: [https://j2damax-serendip-experiential-backend.hf.space](https://j2damax-serendip-experiential-backend.hf.space)
