# Serendip Experiential Backend

FastAPI service providing tourism review classification with explainability features, serving as the API backend for the Serendip Experiential Engine.

## Architecture

This backend uses:
- **FastAPI**: Modern, high-performance web framework for building APIs
- **PyTorch**: Deep learning framework for running the transformer model
- **SHAP**: Machine learning explainability library
- **Hugging Face Transformers**: Pre-trained BERT model fine-tuned for multi-label classification

## Features

- **BERT-based Multi-label Classification**: Analyze reviews across four experiential dimensions
- **SHAP Explainability**: Word-level attribution for model decisions
- **High-performance API**: Optimized for low-latency responses
- **Comprehensive Documentation**: Auto-generated OpenAPI docs

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── analyze.py    # Analysis endpoint
│   │   │   └── health.py     # Health and info endpoints
│   │   ├── dependencies.py   # API dependencies
│   │   └── models.py         # Pydantic data models
│   ├── core/
│   │   ├── config.py         # App configuration
│   │   └── logging.py        # Logging setup
│   ├── ml/
│   │   ├── model.py          # ML model interface
│   │   ├── explainer.py      # SHAP explainer
│   │   └── tokenizer.py      # Text tokenization
│   └── main.py               # Application entry point
├── Dockerfile                # Container definition
└── requirements.txt          # Python dependencies
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze tourism review text, returns scores and explanations |
| `/health` | GET | Check service health status |
| `/info` | GET | Get model information and statistics |
| `/docs` | GET | Interactive API documentation |

### Example Request

```json
POST /analyze
{
  "text": "The sunrise hike through the rainforest was magical. We saw rare birds and our guide explained sustainable tourism practices.",
  "include_explanation": true
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8000 |
| `HOST` | Host to bind | 0.0.0.0 |
| `LOG_LEVEL` | Logging level | info |
| `MODEL_PATH` | Path to model weights | models/bert-tourism-classifier |
| `TOKENIZER_PATH` | Path to tokenizer | models/bert-tokenizer |

## Setup and Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload

# Access API docs at http://localhost:8000/docs
```

### Running with Docker

```bash
# Build container
docker build -t serendip-backend .

# Run container
docker run -p 8000:8000 serendip-backend
```

## Deployment

This service is designed to be deployed on Hugging Face Spaces:

```bash
# Set your Hugging Face token
export HF_TOKEN=your_huggingface_token

# Deploy to Hugging Face
./deploy_to_huggingface.sh
```

## Security Considerations

- Environment variables are used for all sensitive configuration
- CORS is configured to restrict access to authorized domains
- Rate limiting is applied to prevent API abuse

## Live API

The live API is deployed at: [https://j2damax-serendip-experiential-backend.hf.space](https://j2damax-serendip-experiential-backend.hf.space)

Interactive API documentation: [https://j2damax-serendip-experiential-backend.hf.space/docs](https://j2damax-serendip-experiential-backend.hf.space/docs)
