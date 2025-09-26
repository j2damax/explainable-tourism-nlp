# Using the Serendip API Postman Collection

This guide explains how to import and use the Postman collection for testing the Serendip Experiential Engine API.

## Prerequisites

1. [Postman](https://www.postman.com/downloads/) installed on your computer
2. The Serendip Experiential Engine API running (default: http://localhost:8000)

## Importing the Collection

1. Open Postman
2. Click on "Import" in the top left corner
3. Select "File" and choose the `postman_collection.json` file from this directory
4. Click "Import"

## Setting Up Environment Variables

1. Click on "Environments" in the sidebar
2. Click "Create Environment"
3. Name it "Serendip API"
4. Add the following variables:
   - `baseUrl`: `http://localhost:8000` (or your API URL if different)
5. Click "Save"
6. Select the "Serendip API" environment from the dropdown in the top right corner

## Running Tests

### Individual Requests

1. Expand the "Serendip Experiential Engine API" collection
2. Select a request (e.g., "Health Check")
3. Click "Send" to execute the request
4. View the response in the lower panel

### Running All Tests

1. Click on the "..." (three dots) next to the collection name
2. Select "Run collection"
3. In the Collection Runner, configure:
   - Select which requests to run
   - Set the number of iterations
   - Choose the delay between requests
4. Click "Run Serendip Experiential Engine API"
5. View the test results in the runner

## Available Requests

### Health Checks

- **Health Check**: Verify the API is running and model is loaded

### Classification Endpoints

- **Predict - Eco Tourism**: Test eco-tourism review classification
- **Predict - Wellness**: Test wellness review classification
- **Predict - Culinary**: Test culinary review classification
- **Predict - Adventure**: Test adventure review classification

### Explainability Endpoints

- **Explain - Eco Tourism**: Get SHAP explanations for eco-tourism review
- **Explain - Wellness**: Get SHAP explanations for wellness review

### Error Handling Tests

- **Error Test - Empty Review**: Test API response to empty reviews
- **Error Test - Missing Review**: Test API response to missing review field
- **Error Test - Invalid JSON**: Test API response to invalid JSON

## Understanding Test Results

The collection includes automatic tests that verify:

1. Response is valid JSON
2. Response time is under 5 seconds
3. Status codes are appropriate

## Customizing Requests

Feel free to modify the request bodies to test with your own review texts. The variables in the environment can be used to store frequently used values.
