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

## TO DO:
VS Code Prompt	Context / Files to Reference	Outcome (Agent Deliverables)
/new python docker-compose app with two services: a FastAPI backend and a Streamlit frontend. Name the project serendip-experiential-engine. Create separate backend and frontend folders. Ensure the Docker setup runs both services concurrently.	None (Initial setup)	Creates: docker-compose.yml, requirements.txt (root), backend/, frontend/, and basic boilerplate for all files.

Instruction (refining the environment): In the root requirements.txt, ensure you add: fastapi, uvicorn, streamlit, transformers, torch, huggingface-hub, shap, and streamlit-shap.	requirements.txt	Updates requirements.txt with all necessary libraries for the full project scope.

Implement a FastAPI server in /backend/api.py that loads the BertForSequenceClassification model and its tokenizer from the Hugging Face Hub. The model name is j2damax/serendip-travel-classifier. Use AutoModelForSequenceClassification.	#file:backend/api.py	Creates model loading logic.

Define a POST /predict endpoint in /backend/api.py. It must accept a JSON body with a review_text string. It must use the Hugging Face pipeline('text-classification', ...) with function_to_apply="sigmoid" and top_k=None to perform the multi-label classification. Return the results as a JSON array of {'label': str, 'score': float}.	#file:backend/api.py	Creates the main production-ready endpoint.

Add a POST /explain endpoint that accepts a review, runs shap.DeepExplainer or a similar text explainer on the classification pipeline, and returns the HTML string for an interactive SHAP Force Plot visualization.	#file:backend/api.py	Adds the core backend logic for Explainable AI (XAI).

Implement the main Streamlit application in /frontend/app.py. The UI must include a st.title, descriptive text, an st.text_area for user input, and an st.button labeled "Classify & Explain."	#file:frontend/app.py	Creates the basic UI structure.

Implement the classification display logic. After calling the FastAPI's /predict endpoint, convert the JSON output to a pandas DataFrame and display it using both st.dataframe and st.bar_chart to visualize the confidence scores for the four experiential dimensions.	#file:frontend/app.py	Completes the Question 4.a visualization requirement.

Integrate the SHAP explanation. Create a dedicated section below the bar chart. Call the FastAPI's /explain endpoint and use the streamlit-shap component to render the returned HTML of the SHAP Force Plot. Design this section to be intuitive and transparent.	#file:frontend/app.py	Completes the core Question 5.b.ii UI/UX requirement.

Create a function run_genai_benchmark(review_text) in /frontend/genai_module.py. This function must use the OpenAI API to perform a Few-Shot Classification based on the user's review. The prompt must explicitly include 2-3 examples (shots) and instruct the model to return the output in a clean, parseable list format.	New file: /frontend/genai_module.py	Creates the core Few-Shot logic for Question 6.b.

Integrate the GenAI section into /frontend/app.py. Call run_genai_benchmark and display the GPT-4 result alongside the BERT prediction. Then, display a summary table (using st.dataframe) comparing the BERT performance, cost, and reproducibility vs. the GPT-4 performance, cost, and reproducibility to fulfill Question 6.c.	#file:frontend/app.py	Completes the Question 6.c comparative analysis visualization.

Write comprehensive unit tests for the /backend/api.py endpoints (/predict and /explain) using pytest and a mocked model load to ensure the API functions as intended.	#file:backend/api.py	Creates tests for Question 4.b.

Generate a detailed README.md that includes the project overview, the two-part Streamlit/FastAPI architecture diagram, setup instructions (with docker compose up), links to the Hugging Face model, and the key performance metrics (92.50% F1-Score).	@workspace	Creates professional project documentation, fulfilling a key portfolio requirement.