# Explainable Tourism NLP Project

## Multi-Label Classification of Experiential Dimensions in Sri Lankan Tourism Reviews

---

## Presentation Overview

1. Project Introduction
2. Research Objectives
3. Methodology
4. Results & Achievements
5. The Experiential Engine Application
6. Conclusions & Future Work

---

## Project Introduction

### Addressing the "Value Paradox" in Tourism

- Modern travelers seek **experiential value** beyond traditional amenities
- Tourism stakeholders lack tools to **quantify and analyze** these experiential dimensions
- **Unstructured text data** in reviews contains rich insights but is difficult to process at scale
- Need for AI-powered systems that can understand nuanced experiential content

---

## The Four Experiential Dimensions

![Experiential Dimensions](https://placeholder-image.com/experiential-dimensions.jpg)

- **🌱 Regenerative & Eco-Tourism**: Travel focused on social/environmental impact
- **🧘 Integrated Wellness**: Journeys combining physical and mental well-being
- **🍜 Immersive Culinary**: Experiences centered on authentic local cuisine
- **🌄 Off-the-Beaten-Path Adventure**: Exploration of less-crowded landscapes

---

## Research Objectives

1. Develop a **multi-label classification system** for tourism experiential dimensions
2. Create an **explainable AI model** with interpretable outputs for stakeholders
3. Build a **production-ready web application** showcasing the technology
4. Establish a **framework for experiential analytics** in tourism

---

## Project Approach

Two integrated components:

![Project Structure](https://placeholder-image.com/project-structure.jpg)

1. **Tourism Review Classification**: Research-focused model development

   - Data preparation, model training, evaluation, explainability

2. **Serendip Experiential Engine**: Application deployment
   - API development, frontend interface, containerization, cloud hosting

---

## Methodology: Data Collection

- **Dataset**: "Tourism and Travel Reviews: Sri Lankan Destinations" (Mendeley Data)
- **Size**: 16,156 English-language reviews
- **Coverage**: 52 locations across Sri Lanka (2016-2023)
- **Features**: Review text, titles, ratings, locations, user data

![Data Distribution](https://placeholder-image.com/data-distribution.jpg)

---

## Methodology: Data Preprocessing

1. **Text Cleaning**:
   - Lowercase conversion, punctuation removal, normalization
2. **Label Creation**:

   - Semi-supervised approach using keyword lexicons
   - Manual verification of subset for quality control

3. **Data Splitting**:
   - 70% training (12,924 reviews)
   - 15% validation (2,423 reviews)
   - 15% testing (3,232 reviews)

---

## Methodology: Model Development

### Model Selection & Training

- Evaluated multiple transformer architectures:
  - BERT, DistilBERT, RoBERTa, ALBERT
- Selected **BERT base (uncased)** for optimal performance/efficiency
- Adapted for multi-label classification (4 dimensions)
- Fine-tuned with AdamW optimizer and early stopping

![Training Process](https://placeholder-image.com/training-process.jpg)

---

## Methodology: Explainability

### SHAP (SHapley Additive exPlanations)

- Implemented SHAP to provide **word-level explanations** of model predictions
- Identified most influential words for each experiential dimension
- Created visualizations showing positive/negative contribution of key words
- Enhanced model transparency for non-technical stakeholders

![SHAP Example](https://placeholder-image.com/shap-example.jpg)

---

## Results: Model Performance

### Classification Performance

| Dimension                     | F1-Score  | Precision | Recall    |
| ----------------------------- | --------- | --------- | --------- |
| Regenerative & Eco-Tourism    | 0.94      | 0.95      | 0.94      |
| Integrated Wellness           | 0.90      | 0.91      | 0.88      |
| Immersive Culinary            | 0.93      | 0.94      | 0.92      |
| Off-the-Beaten-Path Adventure | 0.92      | 0.93      | 0.91      |
| **Overall**                   | **0.925** | **0.934** | **0.916** |

---

## Results: Model Comparison

| Model           | F1-Score | Training Time | Model Size |
| --------------- | -------- | ------------- | ---------- |
| TF-IDF + LogReg | 0.78     | 3 min         | 25MB       |
| DistilBERT      | 0.90     | 48 min        | 267MB      |
| **BERT base**   | **0.93** | 122 min       | 438MB      |
| RoBERTa base    | 0.93     | 136 min       | 498MB      |
| ALBERT base     | 0.90     | 87 min        | 149MB      |

---

## Results: Key Insights

- **Word Importance**: Context-dependent influence of words across dimensions
  - "organic" → different significance in culinary vs. eco-tourism contexts
- **Geographic Patterns**: Certain locations strongly associated with specific dimensions
  - e.g., Hill Country with wellness, coastal areas with adventure
- **Temporal Trends**: Increasing mentions of regenerative tourism over time

---

## Application: Serendip Experiential Engine

### System Architecture

![System Architecture](https://placeholder-image.com/system-architecture.jpg)

- **Backend**: FastAPI service with model and explainability features
- **Frontend**: Streamlit interface for user interaction
- **Deployment**: Containerized with Docker, hosted on Hugging Face Spaces

---

## Application: User Interface

![User Interface](https://placeholder-image.com/user-interface.jpg)

Key Features:

- Review submission and analysis
- Visualization of dimension scores
- Word-level explanation of predictions
- Comparison with GenAI responses (benchmark)

---

## Application: API Design

RESTful API with three main endpoints:

1. **Health Check** (`GET /`):

   - Verify service status

2. **Prediction** (`POST /predict`):

   - Process review text and return dimension scores

3. **Explanation** (`POST /explain`):
   - Generate SHAP explanations for predictions

---

## Live Demonstration

### Deployed Application:

- **Frontend**: [https://huggingface.co/spaces/j2damax/serendip-experiential-frontend](https://huggingface.co/spaces/j2damax/serendip-experiential-frontend)
- **Backend**: [https://huggingface.co/spaces/j2damax/serendip-experiential-backend](https://huggingface.co/spaces/j2damax/serendip-experiential-backend)

![Demo Screenshot](https://placeholder-image.com/demo-screenshot.jpg)

---

## Challenges & Solutions

1. **Model Loading Performance**:

   - _Challenge_: Slow model initialization in container
   - _Solution_: Lazy loading pattern on first request

2. **SHAP Visualization**:

   - _Challenge_: Converting notebook-style visualizations to web format
   - _Solution_: Custom HTML rendering with base64 image extraction

3. **Cross-Origin Requests**:
   - _Challenge_: Frontend-backend communication
   - _Solution_: CORS middleware implementation

---

## Conclusions

### Project Achievements

1. ✅ Successfully developed high-performance multi-label classifier (F1 = 0.925)
2. ✅ Implemented explainable AI techniques for model transparency
3. ✅ Created production-ready web application with intuitive interface
4. ✅ Deployed containerized solution to cloud platform
5. ✅ Demonstrated practical application of NLP in tourism domain

---

## Research Implications

- **Tourism Product Development**: Data-driven insights for creating experiences
- **Marketing Optimization**: Better targeting based on experiential dimensions
- **Destination Management**: Understanding location-specific experiential strengths
- **Customer Segmentation**: More sophisticated traveler personas
- **Competitive Analysis**: Benchmarking experiential offerings

---

## Future Work

1. **Cross-Cultural Extension**: Apply to different linguistic contexts
2. **Temporal Analysis**: Track evolution of dimensions over time
3. **Multi-Modal Integration**: Incorporate image data with text
4. **Personalization Layer**: User-specific recommendations
5. **Mobile Application**: Native app for on-the-go analysis

---

## Thank You!

### Project Links

- **GitHub Repository**: [github.com/j2damax/explainable-tourism-nlp](https://github.com/j2damax/explainable-tourism-nlp)
- **Live Application**: [huggingface.co/spaces/j2damax/serendip-experiential-frontend](https://huggingface.co/spaces/j2damax/serendip-experiential-frontend)
- **Model**: [huggingface.co/j2damax/serendip-travel-classifier](https://huggingface.co/j2damax/serendip-travel-classifier)

### Contact

- **Email**: j2damax@example.com
- **LinkedIn**: [linkedin.com/in/j2damax](https://linkedin.com/in/j2damax)
