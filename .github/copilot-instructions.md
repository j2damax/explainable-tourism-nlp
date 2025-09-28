# Copilot Instructions for Explainable Tourism NLP Project

## Project Overview

This project is an NLP system for multi-label classification of experiential dimensions in Sri Lankan tourism reviews. It uses transformers to identify four key experiential dimensions: Regenerative & Eco-Tourism, Integrated Wellness, Immersive Culinary, and Off-the-Beaten-Path Adventure.

## Architecture

### Data Flow

1. Raw data (`data/external/mendeley_sri_lanka_reviews.csv`) →
2. Preprocessing (`notebooks/01_eda.ipynb`, `notebooks/02_feature_engineering.ipynb`) →
3. Model training (`notebooks/03_modeling.ipynb`, `notebooks/04_model_training.ipynb`) →
4. Evaluation & Explainability (`notebooks/05_model_evaluation.ipynb`) →
5. Deployment (`notebooks/06_huggingface_deployment.ipynb`)

### Key Components

- `tourism-review-classifier/`: Core package with modular organization
- `data/`: Versioned with DVC (raw → interim → processed)
- `models/`: Trained transformer models
- `reports/figures/`: Generated visualizations for analysis

## Development Workflow

### Environment Setup

```bash
conda env create -f environment.yml
conda activate explainable-tourism-nlp
python verify_installation.py  # Confirms all dependencies are installed
```

### Data Processing

Run the notebook sequence (01-06) or use the CLI interface:

```bash
python -m tourism-review-classifier.dataset  # Process raw data
```

### Model Training

```bash
python -m tourism-review-classifier.modeling.train  # Train model
python -m tourism-review-classifier.modeling.predict  # Run inference
```

### Project Commands (Makefile)

- `make requirements`: Update dependencies
- `make lint`: Check code quality with Ruff
- `make format`: Auto-format code
- `make test`: Run test suite
- `make clean`: Remove compiled files
- `make data`: Process data

## Coding Conventions

### Project Structure

- Cookie-cutter data science template
- Module imports follow absolute imports from `tourism-review-classifier`
- Config paths in `config.py` define the project structure

### Code Patterns

- Use Path objects from `pathlib` for file operations
- Logging through `loguru` with TQDM integration for progress
- CLI interfaces with `typer` for Python modules
- Data versioning with DVC for large files

### Testing

- Tests in `/tests` directory using pytest
- Test data process with `tests/test_data.py`

## Integration Points

### External Dependencies

- Hugging Face ecosystem (transformers, datasets, tokenizers)
- SHAP/LIME for model explainability
- PyTorch for deep learning model training

### Deployment

- Streamlit for interactive web application
- Hugging Face Spaces for model hosting

## Common Tasks

1. Adding new features: Modify `features.py` and update preprocessing in notebooks
2. Training with different models: Update model architecture in `modeling/train.py`
3. Analyzing results: Run evaluation notebook and examine reports/figures
4. Deployment: Follow steps in `06_huggingface_deployment.ipynb`

## Troubleshooting

- Use `verify_installation.py` to confirm all dependencies are installed
- Encoding issues with dataset: Try different encodings (latin1 works)
- CUDA/GPU issues: Check PyTorch installation and CUDA compatibility
- Large file errors: Make sure DVC is properly pulling data files
