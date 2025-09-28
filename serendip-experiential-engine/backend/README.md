---
title: Serendip Experiential Backend
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "3.10"
app_file: api.py
pinned: false
license: mit
---

# Serendip Experiential Backend

This Space hosts the FastAPI backend for the Serendip Experiential Engine, which serves the j2damax/serendip-travel-classifier model.

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Analyzes a tourism review text and returns experiential dimension scores
- `POST /explain`: Provides explainability for prediction results using SHAP

## Usage

This backend API is designed to be used with the [Serendip Experiential Frontend](https://huggingface.co/spaces/j2damax/serendip-experiential-frontend).

## Technologies

- FastAPI
- Hugging Face Transformers
- SHAP for explainability
- PyTorch

## Model

This application uses the `j2damax/serendip-travel-classifier` model, which was trained to identify four key experiential dimensions in Sri Lankan tourism reviews:

- 🌱 Regenerative & Eco-Tourism
- 🧘 Integrated Wellness
- 🍜 Immersive Culinary
- 🌄 Off-the-Beaten-Path Adventure

---

<a href="https://github.com/j2damax/explainable-tourism-nlp" target="_blank">View on GitHub</a>
