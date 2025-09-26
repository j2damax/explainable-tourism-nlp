# Testing the Serendip Experiential Engine API

This document outlines multiple ways to test the Serendip Experiential Engine API endpoints to ensure they are functioning correctly.

## Prerequisites

- The Serendip Experiential Engine services running (start with `./setup.sh`)
- Python 3.8+ (for Python-based testing)
- cURL (for bash script testing)
- [Postman](https://www.postman.com/downloads/) (for Postman collection testing)

## 1. Quick Test with cURL

You can quickly test if the API is working using cURL commands:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint with a sample review
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review_text": "Our eco-lodge in Sri Lanka was sustainable and beautiful."}'

# Test explain endpoint with a sample review
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"review_text": "Our eco-lodge in Sri Lanka was sustainable and beautiful."}'
```

## 2. Using the Bash Test Script

We've provided a bash script for comprehensive API testing:

```bash
# Run all tests
./test_serendip_api.sh

# Test only the health endpoint
./test_serendip_api.sh --endpoint health

# Test only eco-tourism reviews
./test_serendip_api.sh --type eco

# Use a custom API URL
./test_serendip_api.sh --url http://custom-url:8000
```

### Available Options:

- `-e, --endpoint`: Specify which endpoint to test (`health`, `predict`, `explain`, or `all`)
- `-t, --type`: Specify which review type to test (`eco`, `wellness`, `culinary`, `adventure`, `mixed`, or `all`)
- `-u, --url`: Specify a custom base URL for the API
- `-h, --help`: Display help information

## 3. Using the Python Test Script

For a more interactive testing experience with formatted output, use the Python script:

```bash
# Install dependencies first
pip install requests rich

# Run basic test with default options
python test_api.py

# Test specific endpoint
python test_api.py --endpoint explain

# Test with specific review type
python test_api.py --review_type wellness

# Custom review text
python test_api.py --custom_review "Your custom review text here"
```

### Available Options:

- `--endpoint`: Specify which endpoint to test (`health`, `predict`, or `explain`)
- `--review_type`: Specify which example review to use (`eco`, `wellness`, `culinary`, `adventure`, or `mixed`)
- `--custom_review`: Provide your own review text for testing
- `--url`: Specify a custom base URL for the API

## 4. Using Postman

We've included a Postman collection for testing the API:

1. Import `postman_collection.json` into Postman
2. Set up an environment with `baseUrl` variable set to `http://localhost:8000`
3. Run requests individually or use the Collection Runner to run all tests at once

See `postman_testing_guide.md` for detailed instructions on using Postman.

## 5. Testing in the Streamlit UI

The easiest way to test the system end-to-end is through the Streamlit UI:

1. Open http://localhost:8501 in your browser
2. Enter a review in the text area
3. Click "Classify & Explain"
4. Review the classification results and SHAP explanations

## 6. Expected Results

### Health Endpoint

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "j2damax/serendip-travel-classifier"
}
```

### Predict Endpoint

```json
[
  {
    "label": "Regenerative & Eco-Tourism",
    "score": 0.972
  },
  {
    "label": "Integrated Wellness",
    "score": 0.032
  },
  {
    "label": "Immersive Culinary",
    "score": 0.018
  },
  {
    "label": "Off-the-Beaten-Path Adventure",
    "score": 0.017
  }
]
```

### Explain Endpoint

```json
{
  "explanation": "<!DOCTYPE html><html>...</html>"
}
```

## 7. Common Issues & Troubleshooting

1. **Model Loading Timeout**: If the API is slow to respond initially, the model may still be loading. Wait a minute and try again.

2. **Connection Refused**: Check if the Docker containers are running with `docker ps`.

3. **Empty Response**: Check the Docker logs with `docker-compose logs backend`.

4. **CORS Issues**: If testing from a custom frontend, ensure CORS headers are correctly set up.

5. **Memory Errors**: The BERT model requires significant memory. Ensure your system has enough resources available.

For additional help, check the Docker logs or open an issue on the project repository.
