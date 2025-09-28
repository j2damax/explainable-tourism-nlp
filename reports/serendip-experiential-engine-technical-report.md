# Serendip Experiential Engine: Technical Development Report

## Executive Summary

This technical report documents the development of the Serendip Experiential Engine, a web application designed to analyze Sri Lankan tourism reviews and identify key experiential dimensions using natural language processing. The system consists of two main components:

1. A FastAPI backend service that hosts the classification model and provides explainability features
2. A Streamlit frontend that offers an intuitive interface for users to interact with the system

The application successfully demonstrates how transformer-based NLP models can be deployed in a production-ready web application with features for model explainability, making complex AI predictions interpretable for business stakeholders in the tourism industry.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture and Design](#2-architecture-and-design)
3. [Development Process](#3-development-process)
4. [Backend Implementation](#4-backend-implementation)
5. [Frontend Implementation](#5-frontend-implementation)
6. [Deployment Strategy](#6-deployment-strategy)
7. [Technical Challenges and Solutions](#7-technical-challenges-and-solutions)
8. [Conclusion and Future Work](#8-conclusion-and-future-work)

## 1. Project Overview

### 1.1 Business Context

The tourism industry in Sri Lanka is evolving toward more experiential travel offerings. The Serendip Experiential Engine was developed to help tourism stakeholders better understand and categorize visitor experiences through automated analysis of reviews. The system identifies four key experiential dimensions:

- 🌱 **Regenerative & Eco-Tourism**: Travel focused on positive social/environmental impact
- 🧘 **Integrated Wellness**: Journeys combining physical and mental well-being
- 🍜 **Immersive Culinary**: Experiences centered on authentic local cuisine
- 🌄 **Off-the-Beaten-Path Adventure**: Exploration of less-crowded natural landscapes

### 1.2 Project Goals

- Create a production-ready web application that implements the pretrained NLP model
- Provide explainable AI insights for business users
- Enable comparison with generative AI responses for benchmark purposes
- Deploy in a scalable, containerized environment

### 1.3 Technology Stack

- **Backend**: Python, FastAPI, Hugging Face Transformers, PyTorch, SHAP
- **Frontend**: Streamlit, Plotly, HTML/CSS
- **Infrastructure**: Docker, Docker Compose, Hugging Face Spaces
- **Development**: Git, GitHub, VS Code

## 2. Architecture and Design

### 2.1 System Architecture

The application follows a client-server architecture with the following components:

```
[PLACEHOLDER FOR ARCHITECTURE DIAGRAM]
```

#### Key Components:

- **Model Service**: Hosts the transformer-based classification model
- **API Layer**: FastAPI endpoints for prediction and explainability
- **Frontend Application**: Streamlit UI for user interaction
- **Explanation Engine**: SHAP-based visualization and interpretation

### 2.2 Data Flow

1. User submits a tourism review through the Streamlit UI
2. Frontend makes API calls to the backend service
3. Backend processes the text with the transformer model
4. Model outputs prediction scores for four experiential dimensions
5. SHAP explanation is generated to highlight influential words
6. Results are formatted and returned to the frontend
7. Frontend visualizes the predictions and explanations

### 2.3 Container Architecture

The application is containerized using Docker with separate containers for:

- Frontend service (Streamlit)
- Backend service (FastAPI)

These containers are orchestrated with Docker Compose for local development and deployed as standalone containers on Hugging Face Spaces.

## 3. Development Process

### 3.1 Development Environment Setup

The development environment was established using:

```bash
# Clone repository
git clone https://github.com/j2damax/explainable-tourism-nlp.git
cd explainable-tourism-nlp

# Create project structure
mkdir -p serendip-experiential-engine/{frontend,backend}

# Initialize environment files
cp .env.example serendip-experiential-engine/.env
```

### 3.2 Development Workflow

1. **Initial Setup**: Created project structure with separate directories for frontend and backend
2. **Local Development**: Used Docker Compose for local testing and development
3. **Iterative Testing**: Implemented continuous testing of API endpoints and UI components
4. **Containerization**: Created Docker images for both services
5. **Deployment**: Deployed to Hugging Face Spaces for public access

### 3.3 Version Control Strategy

- Feature branches for new functionality
- Pull requests for code review
- Main branch for stable releases
- Development branch for ongoing work

## 4. Backend Implementation

### 4.1 API Design

The backend API follows RESTful principles with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and service status |
| `/predict` | POST | Analyzes review text and returns dimension scores |
| `/explain` | POST | Provides word-level explanations using SHAP |

### 4.2 Model Integration

The backend integrates the `j2damax/serendip-travel-classifier` model from Hugging Face:

```python
# Model initialization code
def load_model():
    """Load model and tokenizer from Hugging Face Hub"""
    global model, tokenizer, classifier
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to use {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=True
    )
```

### 4.3 API Implementation

The backend is implemented using FastAPI with Pydantic models for request/response validation:

```python
class PredictRequest(BaseModel):
    review_text: str

class PredictionResult(BaseModel):
    label: str
    score: float

@app.post("/predict", response_model=List[PredictionResult])
async def predict(request: PredictRequest):
    """
    Classify a tourism review into experiential dimensions
    """
    global model, tokenizer, classifier
    
    # Ensure model is loaded
    if classifier is None:
        await load_model()
    
    # Process input and get predictions
    results = classifier(request.review_text)
    
    # Format response
    predictions = []
    for idx, dimension_scores in enumerate(results):
        for score_data in dimension_scores:
            predictions.append({
                "label": DIMENSIONS[idx],
                "score": score_data["score"]
            })
    
    return predictions
```

### 4.4 Explainability Implementation

The API provides explainability through SHAP (SHapley Additive exPlanations):

```python
@app.post("/explain")
async def explain(request: ExplainRequest):
    """
    Generate SHAP explanations for a prediction
    """
    global model, tokenizer, shap_explainer
    
    # Initialize SHAP explainer if needed
    if shap_explainer is None:
        # Initialize explainer
        shap_explainer = ShapExplainer(model, tokenizer)
    
    # Get explanations
    explanations = shap_explainer.explain_text(
        request.review_text,
        top_n_words=request.top_n_words
    )
    
    return {
        "explanation": explanations
    }
```

### 4.5 Example API Requests and Responses

#### Predict Endpoint

**Request:**
```json
{
  "review_text": "We loved the eco-friendly resort that used solar power and served organic local food. Their conservation efforts were impressive!"
}
```

**Response:**
```json
[
  {
    "label": "Regenerative & Eco-Tourism",
    "score": 0.89
  },
  {
    "label": "Integrated Wellness",
    "score": 0.34
  },
  {
    "label": "Immersive Culinary",
    "score": 0.56
  },
  {
    "label": "Off-the-Beaten-Path Adventure",
    "score": 0.21
  }
]
```

#### Explain Endpoint

**Request:**
```json
{
  "review_text": "We loved the eco-friendly resort that used solar power and served organic local food. Their conservation efforts were impressive!",
  "top_n_words": 5
}
```

**Response:**
```json
{
  "explanation": {
    "top_words": {
      "Regenerative & Eco-Tourism": [
        {"word": "eco-friendly", "value": 0.23, "is_positive": true},
        {"word": "conservation", "value": 0.18, "is_positive": true},
        {"word": "solar", "value": 0.15, "is_positive": true},
        {"word": "organic", "value": 0.12, "is_positive": true},
        {"word": "efforts", "value": 0.08, "is_positive": true}
      ],
      "Immersive Culinary": [
        {"word": "organic", "value": 0.20, "is_positive": true},
        {"word": "food", "value": 0.18, "is_positive": true},
        {"word": "local", "value": 0.12, "is_positive": true},
        {"word": "served", "value": 0.05, "is_positive": true},
        {"word": "loved", "value": 0.03, "is_positive": true}
      ]
    },
    "html": "...[SHAP visualization HTML]..."
  }
}
```

## 5. Frontend Implementation

### 5.1 User Interface Design

The frontend is built with Streamlit and follows a clean, intuitive design:

```
[PLACEHOLDER FOR FRONTEND SCREENSHOT]
```

Key UI components include:
- Text input area for review submission
- Sample reviews for quick testing
- Visualizations of prediction scores
- Interactive word impact analysis
- GenAI comparison section

### 5.2 API Integration

The frontend communicates with the backend through a dedicated API service module:

```python
def analyze_review(review_text: str) -> Optional[Dict[str, Any]]:
    """
    Send review to API for analysis and explanation
    """
    try:
        # Check API health first
        api_status = check_health()
        if not api_status["available"]:
            st.error(api_status["message"])
            return None
            
        # Call predict endpoint
        predict_response = api_call_with_retry(
            requests.post,
            f"{API_URL}/predict",
            json={"review_text": review_text},
            timeout=API_TIMEOUT_LONG
        )
        
        # Call explain endpoint
        explain_response = requests.post(
            f"{API_URL}/explain",
            json={"review_text": review_text, "top_n_words": 5},
            timeout=API_TIMEOUT_SHORT
        )
        
        # Format response
        predictions = predict_response.json()
        result = {
            "predictions": {item["label"]: item["score"] for item in predictions},
            "explanation": explain_response.json().get("explanation", {})
        }
        
        return result
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None
```

### 5.3 Data Visualization

The frontend uses Plotly and Streamlit's native visualization capabilities:

```python
# Create bar chart of dimension scores
dimensions = list(result["predictions"].keys())
scores = list(result["predictions"].values())

# Convert scores to percentages
scores_pct = [round(score * 100, 1) for score in scores]

# Create dataframe for plotting
df_scores = pd.DataFrame({
    'Dimension': dimensions,
    'Score': scores_pct
})

fig = px.bar(
    df_scores,
    y='Dimension',
    x='Score',
    orientation='h',
    color='Score',
    color_continuous_scale='Viridis',
    labels={'Score': 'Confidence Score (%)'},
    text='Score'
)

st.plotly_chart(fig, use_container_width=True)
```

### 5.4 Explainability Visualization

The frontend displays word impact analysis for each dimension:

```
[PLACEHOLDER FOR EXPLAINABILITY VISUALIZATION SCREENSHOT]
```

### 5.5 GenAI Integration

The application includes a GenAI benchmark feature that compares transformer model predictions with OpenAI API responses:

```python
if GENAI_AVAILABLE:
    with st.spinner("Running GPT-4 few-shot classification..."):
        try:
            # Run GenAI benchmark
            genai_results = run_genai_benchmark(review_text)
            
            # Compare with BERT results
            comparison = compare_results(result["predictions"], genai_results)
            
            # Show comparison visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### BERT Classification")
                # Display BERT results
                
            with col2:
                st.markdown(f"### GenAI Classification ({model_name})")
                # Display GenAI results
```

## 6. Deployment Strategy

### 6.1 Docker Containerization

Both frontend and backend are containerized with Docker:

**Frontend Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables
ENV API_URL=http://backend:8000

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Backend Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 Local Deployment with Docker Compose

For local development, Docker Compose orchestrates the services:

```yaml
services:
  backend:
    build: ./backend
    container_name: serendip-backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    environment:
      - PORT=8000
      - HOST=0.0.0.0
      - LOG_LEVEL=info
    networks:
      - serendip-network
    restart: always

  frontend:
    build: ./frontend
    container_name: serendip-frontend
    depends_on:
      - backend
    ports:
      - "8501:8501"
    volumes:
      - ./frontend:/app
    environment:
      - API_URL=http://backend:8000
    networks:
      - serendip-network
    restart: always

networks:
  serendip-network:
    driver: bridge
```

### 6.3 Cloud Deployment to Hugging Face Spaces

The application is deployed to Hugging Face Spaces using custom deployment scripts:

**Frontend Deployment:**
```bash
#!/bin/bash

# Deployment script for Serendip Experiential Frontend to Hugging Face Spaces

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set it with: export HF_TOKEN=your_hugging_face_token"
    exit 1
fi

# Define variables
SPACE_NAME="j2damax/serendip-experiential-frontend"
REPO_URL="https://huggingface.co/spaces/$SPACE_NAME"
LOCAL_DIR="$(pwd)"

# Create a temporary directory for deployment
TMP_DIR=$(mktemp -d)
cd $TMP_DIR

# Initialize git and clone the space if it exists
git init
git config --local user.email "you@example.com"
git config --local user.name "Your Name"

# Clone or create the space
if curl --fail --silent -H "Authorization: Bearer $HF_TOKEN" $REPO_URL > /dev/null; then
    echo "Space exists, cloning repository..."
    git clone "https://huggingface.co/spaces/$SPACE_NAME" .
    git config --local credential.helper store
    echo "https://oauth2:$HF_TOKEN@huggingface.co" > ~/.git-credentials
    find . -mindepth 1 -not -path "./.git*" -delete
else
    echo "Creating new space..."
fi

# Copy files from the frontend directory
cp -r $LOCAL_DIR/* .

# Use the Hugging Face specific Dockerfile
if [ -f "Dockerfile.huggingface" ]; then
    echo "Using Hugging Face specific Dockerfile..."
    mv Dockerfile.huggingface Dockerfile
fi

# Create Hugging Face Space metadata
mkdir -p .hf
cat > .hf/metadata.json << EOL
{
    "app_port": 7860,
    "app_file": "app.py",
    "sdk": "streamlit"
}
EOL

# Push to Hugging Face Spaces
git add .
git commit -m "Deploy frontend to Hugging Face Spaces"
git remote add origin "https://huggingface.co/spaces/$SPACE_NAME"
git push -f origin main

echo "Deployment complete! Your frontend should be available at:"
echo "https://huggingface.co/spaces/$SPACE_NAME"
```

### 6.4 Live Deployment URLs

- **Frontend**: [https://huggingface.co/spaces/j2damax/serendip-experiential-frontend](https://huggingface.co/spaces/j2damax/serendip-experiential-frontend)
- **Backend**: [https://huggingface.co/spaces/j2damax/serendip-experiential-backend](https://huggingface.co/spaces/j2damax/serendip-experiential-backend)

## 7. Technical Challenges and Solutions

### 7.1 Model Loading Performance

**Challenge**: Initial model loading was slow, causing timeout issues during container initialization.

**Solution**: Implemented lazy loading pattern to load the model on first request rather than at startup:

```python
@app.on_event("startup")
async def load_model():
    """Load model and tokenizer from Hugging Face Hub on startup"""
    global model, tokenizer, classifier
    
    logger.info(f"Starting API server. Model will be loaded on first request.")
    # We'll load the model on the first request to avoid timeouts
    pass

# Model is loaded on first prediction
if classifier is None:
    await load_model()
```

### 7.2 Cross-Origin Resource Sharing

**Challenge**: Frontend couldn't access backend API due to CORS restrictions.

**Solution**: Added CORS middleware to the FastAPI application:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 7.3 Streamlit Compatibility in Hugging Face Spaces

**Challenge**: Some Streamlit parameters like `width="stretch"` were incompatible with the version on Hugging Face Spaces.

**Solution**: Updated code to use compatible parameters:

```python
# Before
st.dataframe(df_words, width="stretch", hide_index=True)

# After
st.dataframe(df_words, use_container_width=True, hide_index=True)
```

### 7.4 SHAP Visualization in Web Context

**Challenge**: SHAP visualizations designed for Jupyter notebooks didn't render correctly in the web UI.

**Solution**: Created custom HTML rendering and base64 image extraction:

```python
# Extract base64 image data from HTML
html_content = explanation_data.get("html", "")
if "base64" in html_content:
    try:
        import re
        base64_pattern = re.compile(r'src="data:image/png;base64,([^"]+)"')
        matches = base64_pattern.search(html_content)
        if matches:
            base64_img = matches.group(1)
            result["explanation"]["base64_img"] = base64_img
    except Exception as e:
        print(f"Error extracting base64 image: {str(e)}")
```

### 7.5 Deployment Configuration Issues

**Challenge**: Missing YAML metadata in README.md files caused configuration errors on Hugging Face.

**Solution**: Added proper YAML metadata headers:

```yaml
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
```

## 8. Conclusion and Future Work

### 8.1 Achievements

The Serendip Experiential Engine successfully demonstrates:

1. Production-ready deployment of NLP transformer models in a web application
2. Integration of explainable AI techniques for business users
3. Containerized architecture for scalable deployment
4. Comparison between traditional NLP and generative AI approaches
5. Cloud deployment on Hugging Face Spaces for public access

### 8.2 Future Enhancements

Potential improvements for future versions include:

1. **User Accounts**: Adding authentication and user profiles for saving analyses
2. **Batch Processing**: Supporting bulk analysis of multiple reviews
3. **Advanced Visualizations**: Adding geographical mapping of experiences
4. **Fine-tuning Interface**: Allowing business users to provide feedback to improve model predictions
5. **Multi-language Support**: Expanding to analyze reviews in Sinhala, Tamil, and other languages

### 8.3 Business Impact

The application provides tourism stakeholders with:

1. Data-driven insights into experiential dimensions valued by visitors
2. Understanding of the language and terminology used to describe experiences
3. Ability to quantify and compare different experiential aspects
4. Explainable results that build trust in the AI system

## Appendices

### Appendix A: Environment Variables

```
# Backend environment variables
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=info

# Frontend environment variables
API_URL=http://backend:8000  # For local development
# API_URL=https://j2damax-serendip-experiential-backend.hf.space  # For production

# OpenAI API key for GenAI benchmark comparisons
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

### Appendix B: Requirements

**Frontend Requirements:**
```
streamlit>=1.32.0
pandas
numpy
plotly
requests
python-dotenv
openai>=1.0.0
```

**Backend Requirements:**
```
fastapi
uvicorn
transformers
torch
numpy
shap
python-multipart
```

### Appendix C: Project Structure

```
serendip-experiential-engine/
├── backend/
│   ├── api.py                 # FastAPI application
│   ├── Dockerfile             # Standard Docker configuration
│   ├── Dockerfile.huggingface # HF-specific Docker configuration
│   ├── deploy_to_huggingface.sh # Deployment script
│   └── requirements.txt       # Backend dependencies
├── frontend/
│   ├── app.py                 # Streamlit application
│   ├── api_service.py         # Backend API client
│   ├── config.py              # Configuration settings
│   ├── genai_module.py        # GenAI comparison functionality
│   ├── styles.css             # Custom UI styling
│   ├── Dockerfile             # Standard Docker configuration
│   ├── Dockerfile.huggingface # HF-specific Docker configuration
│   ├── deploy_to_huggingface.sh # Deployment script
│   └── requirements.txt       # Frontend dependencies
├── docker-compose.yml         # Local Docker orchestration
└── .env                       # Environment variables
```

---

*This report was prepared for academic submission and documents the technical aspects of the Serendip Experiential Engine project.*