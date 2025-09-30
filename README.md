# Serendip Travel - Explainable Tourism NLP

This repository contains two related projects focused on analyzing and classifying tourism experiences:

## 1. Tourism Review Classification

Research project that developed an NLP model for classifying Sri Lankan tourist reviews into four experiential dimensions.

- Data analysis and model training
- BERT-based multi-label classification
- Explainable AI using SHAP

[Tourism Review Classification Project](./tourism-review-classification/README.md)

## 2. Serendip Experiential Engine

Web application that implements the tourism classification model with a user-friendly interface.

### Live Demo

- [Frontend Application](https://huggingface.co/spaces/j2damax/serendip-experiential-frontend)
- [Backend API](https://huggingface.co/spaces/j2damax/serendip-experiential-backend)

### Experiential Dimensions

- 🌱 **Regenerative & Eco-Tourism**: Travel focused on positive social/environmental impact
- 🧘 **Integrated Wellness**: Journeys combining physical and mental well-being
- 🍜 **Immersive Culinary**: Experiences centered on authentic local cuisine
- 🌄 **Off-the-Beaten-Path Adventure**: Exploration of less-crowded natural landscapes

### Quick Start

```bash
# Run locally
cd serendip-experiential-engine
cp .env.example .env
# Edit .env to add OPENAI_API_KEY
docker compose up

# Deploy to Hugging Face
cd serendip-experiential-engine
export HF_TOKEN=your_huggingface_token
./deploy_all.sh
```

- Frontend: [https://huggingface.co/spaces/j2damax/serendip-experiential-frontend](https://huggingface.co/spaces/j2damax/serendip-experiential-frontend)
- Backend: [https://huggingface.co/spaces/j2damax/serendip-experiential-backend](https://huggingface.co/spaces/j2damax/serendip-experiential-backend)

[View Serendip Experiential Engine Project](./serendip-experiential-engine/README.md)

## Repository Structure

```
.
├── tourism-review-classification/  # Original research project
│   ├── data/                      # Dataset files
│   ├── models/                    # Trained models
│   ├── notebooks/                 # Research notebooks
│   └── ...                        # Other research materials
│
└── serendip-experiential-engine/  # Web application
    ├── backend/                   # FastAPI backend
    ├── frontend/                  # Streamlit frontend
    └── ...                        # Docker and deployment files
```

## Getting Started

### Tourism Review Classification

Follow the setup instructions in the [Tourism Review Classification README](./tourism-review-classification/README.md).

### Serendip Experiential Engine

1. Navigate to the web application directory:

```bash
cd serendip-experiential-engine
```

2. Start the application using Docker:

```bash
docker-compose up
```

3. Access the frontend at http://localhost:8501

# Create virtual environment

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies

pip install -r requirements.txt

# Verify installation

python verify_installation.py

````

### Running the EDA

```bash
# Start Jupyter Lab
jupyter lab

# Open tourism_data_cleaning_and_entity_linking.ipynb
# Run all cells to perform exploratory data analysis
````

### Running the Web Application

```bash
# Start the Streamlit app
streamlit run app.py
```

## Dataset

The project uses the "Tourism and Travel Reviews: Sri Lankan Destinations" dataset from Mendeley Data, containing 16,156 English-language reviews of Sri Lankan destinations.

**Dataset Features:**

- Review text and titles
- Location information (cities, types)
- User demographics and ratings
- Temporal data

## 🏗️ Project Structure

```
├── LICENSE                    <- MIT License
├── Makefile                   <- Convenience commands
├── README.md                  <- This file
├── environment.yml            <- Conda environment specification
├── requirements.txt           <- Pip requirements
├── pyproject.toml            <- Project configuration
├── data/
│   ├── external/             <- Original dataset (mendeley_sri_lanka_reviews.csv)
│   ├── interim/              <- Processed data
│   └── processed/            <- Final datasets for modeling
├── notebooks/
│   ├── 01_comprehensive_eda.ipynb           <- Exploratory Data Analysis
│   ├── 02_data_preprocessing.ipynb          <- Data cleaning and preprocessing
│   ├── 03_model_training.ipynb              <- Model training and evaluation
│   └── 04_explainable_ai.ipynb              <- SHAP explanations
├── tourism-review-classifier/ <- Source code
│   ├── config.py             <- Configuration settings
│   ├── dataset.py            <- Dataset handling
│   ├── features.py           <- Feature engineering
│   ├── modeling/
│   │   ├── train.py          <- Model training
│   │   └── predict.py        <- Model inference
│   └── plots.py              <- Visualization utilities
├── models/                   <- Trained models
├── reports/                  <- Generated reports and figures
└── docs/                     <- Documentation
```

## Key Technologies

### Core ML/NLP

- **Transformers**: Hugging Face ecosystem for fine-tuning
- **PyTorch**: Deep learning framework
- **SHAP**: Explainable AI for model interpretability

### Web Deployment

- **Streamlit**: Interactive web application
- **FastAPI**: Backend API (optional)
- **Hugging Face Spaces**: Deployment platform

### Development

- **DVC**: Data version control
- **Ruff**: Code linting and formatting
- **Pytest**: Testing framework

## Usage Examples

### Running EDA

```python
# In Jupyter notebook
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/external/mendeley_sri_lanka_reviews.csv', encoding='latin1')

# Perform analysis (see notebooks/01_comprehensive_eda.ipynb)
```

### Model Training

```python
from tourism_review_classifier.modeling.train import train_model
from tourism_review_classifier.config import Config

# Train the model
model = train_model(Config())
```

### Making Predictions

```python
from tourism_review_classifier.modeling.predict import predict_review

# Classify a review
review_text = "Amazing eco-friendly resort with authentic local cuisine"
predictions = predict_review(review_text)
print(predictions)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: Tourism and Travel Reviews: Sri Lankan Destinations (Mendeley Data)
- **Framework**: Cookiecutter Data Science Project Template

## Contact

For questions about this project, please contact:

- **Author**: B M J N Balasuriya
- **Email**: COMScDS242P-009@student.nibm.lk/j2damax@gmail.com

---
