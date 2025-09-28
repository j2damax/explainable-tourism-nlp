# Building an Explainable AI System for Tourism Experience Classification

*How we developed a transformer-based NLP system to understand and quantify experiential dimensions in tourism reviews*

![Header Image: Tourism experiences in Sri Lanka with AI visualization overlay](https://placeholder-image.com/header-image.jpg)

## Introduction

In the age of experiential travel, tourists are increasingly seeking authentic, meaningful experiences rather than just comfortable accommodations or picturesque views. They want to immerse themselves in local cultures, contribute to sustainable tourism, pursue wellness, and discover hidden gems off the beaten path.

However, tourism stakeholders face a significant challenge: How do you quantify these abstract experiential dimensions at scale? Traditional analytics tools can count star ratings and identify basic sentiment, but they struggle to understand the nuanced experiential content embedded in reviews.

In this article, I'll walk through how we tackled this problem by developing an explainable AI system that can classify Sri Lankan tourism reviews into four key experiential dimensions. The project combines cutting-edge NLP with practical business applications, culminating in a web application that tourism businesses can use to gain actionable insights.

## The Value Paradox in Tourism

Before diving into the technical implementation, let's understand the problem we're trying to solve: what I call the "Value Paradox" in tourism.

Modern travelers actively seek experiences aligned with their values and interests—whether that's sustainability, wellness, culinary adventures, or exploring off-the-beaten-path locations. However, tourism analytics remains heavily focused on conventional metrics like occupancy rates, average daily rates, and simplistic satisfaction scores.

This creates an information gap where the most valuable data—the rich, descriptive content in reviews describing experiential qualities—remains largely untapped because it's unstructured and difficult to analyze at scale.

Our solution addresses this paradox by using transformer models to identify four key experiential dimensions:

1. **🌱 Regenerative & Eco-Tourism**: Travel focused on positive social/environmental impact
2. **🧘 Integrated Wellness**: Journeys combining physical and mental well-being
3. **🍜 Immersive Culinary**: Experiences centered on authentic local cuisine
4. **🌄 Off-the-Beaten-Path Adventure**: Exploration of less-crowded natural landscapes

## Project Architecture Overview

The complete solution consists of two main components:

1. **Tourism Review Classification**: The research-focused model development pipeline
2. **Serendip Experiential Engine**: The production web application

![System Architecture Diagram](https://placeholder-image.com/architecture.jpg)

Let's explore each of these components in detail.

## Part 1: Building the Classification Model

### Data Collection and Preparation

We used the "Tourism and Travel Reviews: Sri Lankan Destinations" dataset from Mendeley Data, containing 16,156 English-language reviews. This comprehensive dataset includes reviews across 52 locations in Sri Lanka, complete with ratings, location types, and user demographics.

The first challenge was creating labeled data for training our model. Since we were identifying abstract experiential dimensions not explicitly labeled in the original dataset, we developed a semi-supervised labeling approach:

1. Created keyword lexicons for each experiential dimension (e.g., "sustainable," "eco-friendly," "conservation" for Regenerative Tourism)
2. Used these lexicons to assign initial labels based on keyword frequency
3. Manually verified a subset of these labels to ensure quality
4. Split the dataset into training (70%), validation (15%), and test (15%) sets

```python
# Example of our keyword-based labeling approach
def assign_initial_labels(text, keyword_lists):
    """Assign initial labels based on keyword presence"""
    labels = []
    for dimension, keywords in keyword_lists.items():
        # Count keyword occurrences
        count = sum(1 for keyword in keywords if keyword in text.lower())
        # Apply threshold-based labeling
        labels.append(1 if count >= THRESHOLD else 0)
    return labels
```

### Model Selection and Training

After experimenting with various transformer architectures, we selected BERT base (uncased) as our primary model, balancing performance with computational efficiency.

The model was adapted for multi-label classification by modifying the output layer to produce independent probabilities for each dimension. We fine-tuned it using the AdamW optimizer with a linear learning rate scheduler and implemented early stopping based on validation F1-score.

```python
# Model initialization for multi-label classification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4,
    problem_type="multi_label_classification"
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

### Model Evaluation

Our final model achieved impressive performance across all four dimensions:

| Dimension | F1-Score | Precision | Recall |
|-----------|----------|-----------|--------|
| Regenerative & Eco-Tourism | 0.94 | 0.95 | 0.94 |
| Integrated Wellness | 0.90 | 0.91 | 0.88 |
| Immersive Culinary | 0.93 | 0.94 | 0.92 |
| Off-the-Beaten-Path Adventure | 0.92 | 0.93 | 0.91 |
| **Overall** | **0.93** | **0.93** | **0.92** |

These results demonstrate that transformer models can effectively identify abstract experiential dimensions in tourism text with high accuracy.

### Implementing Explainability

A critical requirement was making the model's decisions interpretable for business stakeholders who may not have a technical background. For this, we implemented SHAP (SHapley Additive exPlanations) to provide word-level insights into what drives predictions.

```python
class ShapExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def explain_text(self, text, top_n_words=10):
        """Generate SHAP explanations for a given text"""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Create explainer and generate explanations
        explainer = shap.Explainer(self.predict_fn, self.tokenizer)
        shap_values = explainer(text)
        
        # Process and return word-level explanations
        return self._process_shap_values(shap_values, text, top_n_words)
```

SHAP analysis provided fascinating insights into how the model "thinks." For example, we discovered that words like "organic" have different significance depending on whether they appear in culinary or eco-tourism contexts.

![SHAP Visualization Example](https://placeholder-image.com/shap-example.jpg)

## Part 2: Developing the Experiential Engine Application

With our model trained and evaluated, we shifted focus to building a production-ready web application that tourism stakeholders could actually use.

### Backend Development with FastAPI

We developed a FastAPI backend service to host the model and provide API endpoints for prediction and explainability. The service follows RESTful principles with three main endpoints:

1. **Health Check** (`GET /`): Verifies service status
2. **Prediction** (`POST /predict`): Analyzes review text and returns dimension scores
3. **Explanation** (`POST /explain`): Provides word-level explanations using SHAP

```python
@app.post("/predict", response_model=List[PredictionResult])
async def predict(request: PredictRequest):
    """Classify a tourism review into experiential dimensions"""
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

A key challenge we faced was model loading performance, as the initial loading of transformer models can be slow. We solved this with a lazy loading pattern that initializes the model on the first request rather than at startup, avoiding timeout issues during container initialization.

### Frontend with Streamlit

For the frontend, we chose Streamlit for its rapid development capabilities and straightforward integration with data visualization libraries. The interface is clean and intuitive, focusing on the core user journey:

1. Enter a tourism review or select from sample reviews
2. View predicted experiential dimension scores with visualizations
3. Explore word-level explanations for each dimension

![Frontend Screenshot](https://placeholder-image.com/frontend-screenshot.jpg)

One interesting feature we added was a GenAI comparison module that allows users to benchmark our model's predictions against responses from a large language model like GPT-4. This provides valuable context and helps users understand the advantages and limitations of our specialized model compared to general-purpose AI.

### Containerization and Deployment

Both frontend and backend components were containerized using Docker for consistent environment management:

```dockerfile
# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

For local development, we used Docker Compose to orchestrate the services, while for production deployment, we chose Hugging Face Spaces for its specialized ML deployment capabilities and ease of use.

The deployment process included custom scripts to handle the specific requirements of Hugging Face Spaces, particularly around metadata configuration and port management.

## Technical Challenges and Solutions

Throughout the project, we encountered several technical challenges worth highlighting:

### 1. Cross-Origin Resource Sharing

**Challenge**: The frontend couldn't access the backend API due to CORS restrictions.

**Solution**: We added CORS middleware to the FastAPI application to allow cross-origin requests:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. SHAP Visualization in Web Context

**Challenge**: SHAP visualizations designed for Jupyter notebooks didn't render correctly in the web UI.

**Solution**: We created custom HTML rendering with base64 image extraction:

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

### 3. Streamlit Compatibility Issues

**Challenge**: Some Streamlit parameters were incompatible with the version on Hugging Face Spaces.

**Solution**: We updated code to use compatible parameters:

```python
# Before
st.dataframe(df_words, width="stretch", hide_index=True)

# After
st.dataframe(df_words, use_container_width=True, hide_index=True)
```

## Business Applications and Impact

The system we built has several practical applications for tourism stakeholders:

### Data-Driven Product Development

Tourism businesses can use the dimensional analysis to identify their experiential strengths and gaps. For example, a resort might discover they score highly on Wellness and Culinary dimensions but need to improve their Eco-Tourism offerings to appeal to environmentally conscious travelers.

### Marketing Optimization

By understanding which experiential dimensions resonate most with their visitors, businesses can tailor their marketing messaging accordingly. A hotel that consistently receives high scores for "Off-the-Beaten-Path Adventure" can emphasize these unique experiences in their marketing materials.

### Competitive Analysis

The system enables businesses to analyze competitor reviews and benchmark their experiential offerings against others in their region or category.

### Customer Segmentation

Beyond traditional demographic segmentation, businesses can develop more sophisticated traveler personas based on experiential preferences.

## Future Directions

While our current system has proven effective, there are several exciting avenues for future development:

### 1. Multi-language Support

Extending the model to analyze reviews in Sinhala, Tamil, and other languages would provide a more comprehensive understanding of visitor experiences, especially from domestic tourists.

### 2. Temporal Analysis

Implementing tools to track how experiential dimensions evolve over time would help businesses identify emerging trends and adapt their offerings accordingly.

### 3. Fine-grained Classification

Developing more nuanced sub-dimensions for each major category would provide even more detailed insights into specific aspects of visitor experiences.

### 4. Mobile Application

Creating a native mobile application would allow tourism businesses to analyze reviews on-the-go and receive real-time insights.

## Conclusion

Our project demonstrates how cutting-edge NLP techniques can be applied to solve practical business challenges in the tourism industry. By developing a system that can understand and quantify abstract experiential dimensions, we've helped bridge the "Value Paradox" gap between what modern travelers seek and what traditional analytics can measure.

The combination of high-performance classification with explainable outputs makes this technology particularly valuable for business stakeholders who need actionable insights without requiring technical expertise in machine learning.

As tourism continues to evolve toward more personalized, experience-focused offerings, AI systems like ours will play an increasingly important role in helping businesses understand and meet visitor expectations in a data-driven way.

---

### Try It Yourself

The Serendip Experiential Engine is available for public use at:
- Frontend: [https://huggingface.co/spaces/j2damax/serendip-experiential-frontend](https://huggingface.co/spaces/j2damax/serendip-experiential-frontend)
- Backend: [https://huggingface.co/spaces/j2damax/serendip-experiential-backend](https://huggingface.co/spaces/j2damax/serendip-experiential-backend)

For developers interested in the technical implementation, the complete code is available on GitHub: [github.com/j2damax/explainable-tourism-nlp](https://github.com/j2damax/explainable-tourism-nlp)

---

*About the Author: [Your bio here]*