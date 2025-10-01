# Serendip Travel - Explainable Tourism NLP

An AI-powered system for analyzing experiential dimensions in tourism reviews with explainable results and interactive visualizations.

## Project Overview

This repository contains two integrated components:

1. **Tourism Review Classification** - Research-focused ML project that trains and evaluates BERT-based classification models on Sri Lankan tourism reviews
2. **Serendip Experiential Engine** - Production web application with an intuitive interface for analyzing tourism reviews in real-time

## Project Structure

```
explainable-tourism-nlp/
├── tourism-review-classification/  # Research ML component
│   ├── data/                       # Tourism review datasets (DVC managed)
│   ├── notebooks/                  # Analysis and training notebooks
│   ├── models/                     # Trained BERT models
│   └── reports/                    # Evaluation results and figures
└── serendip-experiential-engine/   # Production web application
    ├── backend/                    # FastAPI service with model serving
    └── frontend/                   # Streamlit interactive interface
```

## Experiential Dimensions

The system analyzes tourism reviews across four key experiential dimensions:

- 🌱 **Regenerative & Eco-Tourism**: Sustainable travel experiences with positive environmental and social impact
- 🧘 **Integrated Wellness**: Experiences focused on physical and mental well-being
- 🍜 **Immersive Culinary**: Authentic local cuisine and food-related experiences
- 🌄 **Off-the-Beaten-Path Adventure**: Exploration of less-crowded natural areas and unique activities

## Features

- **Multi-label Classification**: Analyzes reviews across all four dimensions simultaneously
- **Explainable AI**: SHAP-based word-level explanations for model decisions
- **Interactive Visualization**: Intuitive radar charts and word highlighting
- **GenAI Comparison**: Optional comparison with GPT responses for validation
- **High Accuracy**: 92% F1-score on multi-label classification tasks

## Getting Started

### Local Setup

```bash
# Clone the repository
git clone https://github.com/j2damax/explainable-tourism-nlp.git
cd explainable-tourism-nlp/serendip-experiential-engine

# Configure environment
cp .env.example .env
# Edit .env to add OPENAI_API_KEY

# Run with Docker
docker compose up
```

### Live Demo

- **Frontend**: [Serendip Experiential Frontend](https://huggingface.co/spaces/j2damax/serendip-experiential-frontend) - Interactive web interface
- **Backend API**: [Serendip Experiential Backend](https://huggingface.co/spaces/j2damax/serendip-experiential-backend) - REST API with model serving

## Component Documentation

For detailed documentation of each component:

- [Tourism Review Classification](./tourism-review-classification/README.md) - Research project documentation
- [Serendip Experiential Engine](./serendip-experiential-engine/README.md) - Web application documentation
  - [Backend API](./serendip-experiential-engine/backend/README.md) - FastAPI service
  - [Frontend](./serendip-experiential-engine/frontend/README.md) - Streamlit application

## Usage

### Tourism Review Classification

For model training and research:

```bash
# Navigate to the research project
cd tourism-review-classification

# Create environment
conda env create -f environment.yml
conda activate explainable-tourism-nlp

# Verify installation
python verify_installation.py
```

### Serendip Experiential Engine

For running the web application:

```bash
# Navigate to the web application
cd serendip-experiential-engine

# Start with Docker
docker-compose up

# Access the frontend at http://localhost:8501
```

## Dataset

The project uses the "Tourism and Travel Reviews: Sri Lankan Destinations" dataset from Mendeley Data, containing 16,156 English-language reviews of Sri Lankan tourist destinations.

**Dataset Features:**

- Review text and titles
- Location information (cities, attraction types)
- User demographics and ratings (1-5 stars)
- Temporal data (visit dates, seasons)

## Technologies

### Machine Learning & NLP

- **Hugging Face Transformers**: BERT model fine-tuning and inference
- **PyTorch**: Deep learning framework for model training
- **SHAP**: Explainable AI for word-level feature importance
- **scikit-learn**: Evaluation metrics and preprocessing

### Web Development

- **Streamlit**: Interactive web interface with visualizations
- **FastAPI**: High-performance REST API for model serving
- **Docker**: Containerization for consistent deployment
- **Hugging Face Spaces**: Hosting platform for both components

## Development Tools

- **DVC**: Data version control for large files
- **Ruff & Black**: Code linting and formatting
- **Pytest**: Test automation
- **MkDocs**: Documentation generation

## Project Organization

```
explainable-tourism-nlp/
│
├── tourism-review-classification/   # Research component
│   ├── data/                        # Datasets managed by DVC
│   │   ├── external/                # Original dataset files
│   │   ├── interim/                 # Intermediate processed data
│   │   └── processed/               # Final datasets for modeling
│   │
│   ├── models/                      # Trained model files
│   ├── notebooks/                   # Analysis notebooks
│   │   ├── 01_eda.ipynb             # Data exploration
│   │   ├── 02_feature_engineering.ipynb  # Feature preparation
│   │   ├── 03_modeling.ipynb        # Model selection
│   │   ├── 04_model_training.ipynb  # BERT training
│   │   ├── 05_model_evaluation.ipynb  # Evaluation metrics
│   │   └── 06_huggingface_deployment.ipynb  # Deployment
│   │
│   └── reports/                     # Generated analysis reports
│       └── figures/                 # Generated graphics and plots
│
└── serendip-experiential-engine/    # Web application
    ├── backend/                     # FastAPI service
    │   ├── app/                     # Application code
    │   ├── Dockerfile               # Container configuration
    │   └── requirements.txt         # Python dependencies
    │
    ├── frontend/                    # Streamlit interface
    │   ├── app.py                   # Main application
    │   ├── components/              # UI components
    │   └── utils/                   # Utility functions
    │
    └── docker-compose.yml           # Multi-container setup
```

## Example Usage

### Research Component

```python
# Using the trained model for prediction
from tourism_review_classification.modeling.predict import predict

review_text = "We stayed at an eco-friendly resort surrounded by nature. They use solar power and have a zero-waste policy. The yoga classes each morning were rejuvenating, and we enjoyed authentic local cuisine made with ingredients from their organic garden."

predictions = predict(review_text)
print(predictions)

# Output:
# {
#    'regenerative_eco_tourism': 0.92,
#    'integrated_wellness': 0.78,
#    'immersive_culinary': 0.85,
#    'off_beaten_path': 0.45
# }
```

### Web Application

The Streamlit interface allows:

1. **Input Review Text**: Enter or paste any tourism review
2. **View Predictions**: See scores for each experiential dimension
3. **Explore Explanations**: Understand which words influenced each prediction
4. **Compare with GenAI**: (Optional) View comparison with OpenAI's analysis

## Citing This Work

If you use this project in your research, please cite it using:

```bibtex
@software{balasuriya2023serendip,
  author = {Balasuriya, B M J N},
  title = {Explainable Tourism NLP: Multi-label Classification of Experiential Dimensions},
  year = {2023},
  url = {https://github.com/j2damax/explainable-tourism-nlp}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: Tourism and Travel Reviews: Sri Lankan Destinations (Mendeley Data)
- **Infrastructure**: Hugging Face Spaces for deployment hosting
- **Framework**: Cookiecutter Data Science Project Template

## Contact

For questions or feedback about this project:

- **Author**: B M J N Balasuriya
- **Email**: j2damax@gmail.com

---
