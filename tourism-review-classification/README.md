# Tourism Review Classification

Research project for multi-label classification of experiential dimensions in Sri Lankan tourism reviews using transformer-based NLP models.

## Project Overview

This project implements a machine learning system that analyzes tourism reviews and classifies them into four key experiential dimensions:

- **Regenerative & Eco-Tourism**: Sustainable travel experiences with positive environmental and social impact
- **Integrated Wellness**: Experiences focused on physical and mental well-being
- **Immersive Culinary**: Authentic local cuisine and food-related experiences
- **Off-the-Beaten-Path Adventure**: Exploration of less-crowded natural areas and unique activities

The system uses state-of-the-art transformer models (BERT) to achieve high accuracy in multi-label classification, with explainability features to understand model decisions.

## Project Structure

```
├── data/               # Data files versioned with DVC
│   ├── external/       # External source data
│   ├── interim/        # Intermediate processed data
│   └── processed/      # Final processed datasets
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for analysis
├── reports/            # Generated reports and figures
│   └── figures/        # Generated graphics and plots
├── docs/               # Project documentation
└── references/         # Explanatory materials and references
```

## Setup Instructions

### Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate explainable-tourism-nlp

# Verify installation
python verify_installation.py

# Pull data using DVC (if not already present)
dvc pull
```

### Development Workflow

The project follows a structured workflow through Jupyter notebooks:

1. **Data Exploration**: `notebooks/01_eda.ipynb`
2. **Feature Engineering**: `notebooks/02_feature_engineering.ipynb` 
3. **Model Selection**: `notebooks/03_modeling.ipynb`
4. **Model Training**: `notebooks/04_model_training.ipynb`
5. **Model Evaluation**: `notebooks/05_model_evaluation.ipynb`
6. **Deployment**: `notebooks/06_huggingface_deployment.ipynb`

### Makefile Commands

```bash
make requirements   # Update dependencies
make lint           # Check code quality with Ruff
make format         # Auto-format code
make clean          # Remove compiled files
```

## Key Results and Performance

- **92% F1-score** for multi-label classification across all four dimensions
- **SHAP-based model explanation** providing insight into feature importance
- **Deployed model** available through Hugging Face Spaces
- **Interactive visualization** of model predictions and explanations

## Dependencies

Key dependencies include:
- PyTorch and Hugging Face Transformers for model training
- SHAP for model explainability
- Streamlit and FastAPI for deployment
- Data versioning with DVC

## Deployment

The model is deployed as a public API and web application on Hugging Face Spaces. For deployment instructions, refer to `notebooks/06_huggingface_deployment.ipynb`.

## Troubleshooting

- For dataset encoding issues, try using 'latin1' encoding
- CUDA/GPU issues: Verify PyTorch installation and CUDA compatibility
- Missing data files: Run `dvc pull` to retrieve large files
