#!/bin/bash

# Simplified script for deploying backend to Hugging Face Spaces

# Exit on error
set -e

# Set up directories
SPACE_NAME="j2damax/serendip-experiential-backend"
BACKEND_DIR="/Users/jam/serendip-travel/explainable-tourism-nlp/serendip-experiential-engine/backend"
TEMP_DIR="/tmp/serendip-backend-deploy"

echo "🚀 Starting deployment process..."

# Ask for Hugging Face token
echo "Please enter your Hugging Face access token (from https://huggingface.co/settings/tokens):"
read -s HF_TOKEN
echo ""

# Create and clean temporary directory
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR
echo "📁 Created temporary directory"

# Clone the Hugging Face Space repository using token
echo "📥 Cloning Hugging Face Space repository..."
git clone https://$HF_TOKEN@huggingface.co/spaces/$SPACE_NAME $TEMP_DIR

# Copy your backend files to the temp directory
echo "📋 Copying backend files..."
cp -r $BACKEND_DIR/* $TEMP_DIR/

# Use the correct Dockerfile
echo "🐳 Setting up Dockerfile..."
cp $BACKEND_DIR/Dockerfile.huggingface $TEMP_DIR/Dockerfile

# Create a simple .gitignore file to avoid committing unnecessary files
echo "__pycache__/" > $TEMP_DIR/.gitignore
echo "*.pyc" >> $TEMP_DIR/.gitignore
echo ".DS_Store" >> $TEMP_DIR/.gitignore

# Navigate to the temporary directory
cd $TEMP_DIR

# Add, commit, and push changes
echo "💾 Committing changes..."
git add .
git commit -m "Deploy backend from local repository"

echo "📤 Pushing to Hugging Face Spaces..."
git push

echo "✅ Deployment completed!"
echo "Your backend should be available at: https://huggingface.co/spaces/$SPACE_NAME"
echo "It might take a few minutes for the Space to build and deploy."
echo "It might take a few minutes to build and deploy."

# Return to the original directory
cd -