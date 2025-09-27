# Tourism Review Classification

This directory contains the code and resources for the Tourism Review Classification project, which develops an explainable NLP model to classify experiential dimensions in Sri Lankan tourist reviews.

## Project Overview

This project addresses the "Value Paradox" in Sri Lankan tourism by developing an AI system that can understand and quantify abstract experiential dimensions from tourist reviews. The system classifies reviews into four key dimensions:

- **Regenerative & Eco-Tourism**: Travel focused on positive social/environmental impact
- **Integrated Wellness**: Journeys combining physical and mental well-being  
- **Immersive Culinary**: Experiences centered on authentic local cuisine
- **Off-the-Beaten-Path Adventure**: Exploration of less-crowded natural landscapes

## Directory Structure

- `data/`: Contains the raw, interim, and processed data files
- `models/`: Trained model files and model artifacts
- `notebooks/`: Jupyter notebooks for exploratory data analysis, model training, and evaluation
- `reports/`: Generated analysis and visualizations
- `docs/`: Documentation files
- `references/`: Reference materials and literature

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- Conda (recommended) or pip

### Installation

```bash
# Navigate to the project directory
cd tourism-review-classification

# Create and activate conda environment
conda env create -f environment.yml
conda activate explainable-tourism-nlp

# Verify installation
python verify_installation.py
```

## Usage

The project consists of several Jupyter notebooks that should be run in sequence:

1. `notebooks/01_eda.ipynb`: Exploratory data analysis
2. `notebooks/02_feature_engineering.ipynb`: Feature engineering
3. `notebooks/03_modeling.ipynb`: Model selection
4. `notebooks/04_model_training.ipynb`: Model training
5. `notebooks/05_model_evaluation.ipynb`: Model evaluation
6. `notebooks/06_huggingface_deployment.ipynb`: Deployment to Hugging Face

## Makefile Commands

- `make requirements`: Update dependencies
- `make lint`: Check code quality
- `make format`: Auto-format code
- `make test`: Run test suite
- `make clean`: Remove compiled files
- `make data`: Process data