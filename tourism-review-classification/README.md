# Tourism Review Classification

Research project for classifying experiential dimensions in tourism reviews.

## Project Overview

This project uses NLP to classify tourism reviews into four key dimensions:

- **Regenerative & Eco-Tourism**: Sustainable travel with positive impact
- **Integrated Wellness**: Physical and mental well-being experiences
- **Immersive Culinary**: Authentic local cuisine experiences
- **Off-the-Beaten-Path Adventure**: Exploring less-crowded natural areas

## Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate explainable-tourism-nlp

# Verify installation
python verify_installation.py
```

## Key Notebooks

1. `notebooks/01_eda.ipynb`: Data exploration
2. `notebooks/02_feature_engineering.ipynb`: Feature preparation
3. `notebooks/03_modeling.ipynb`: Model selection
4. `notebooks/04_model_training.ipynb`: BERT model training
5. `notebooks/05_model_evaluation.ipynb`: Evaluation and explainability
6. `notebooks/06_huggingface_deployment.ipynb`: Deployment

## Results

- 92% F1-score for multi-label classification
- SHAP-based explanation of model decisions
- Model deployed via Hugging Face Spaces
