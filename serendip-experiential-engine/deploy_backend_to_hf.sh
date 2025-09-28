#!/bin/bash

# This script deploys the backend to Hugging Face Spaces

# Exit on error
set -e

# Variables
SPACE_NAME="j2damax/serendip-experiential-backend"
BACKEND_DIR="/Users/jam/serendip-travel/explainable-tourism-nlp/serendip-experiential-engine/backend"
TEMP_DIR="/tmp/serendip-backend-deploy"

echo "🚀 Deploying backend to Hugging Face Spaces: $SPACE_NAME"

# Create a temporary directory
mkdir -p $TEMP_DIR
echo "📁 Created temporary directory at $TEMP_DIR"

# Clone the Hugging Face Space repository
echo "📥 Cloning Hugging Face Space repository..."
git clone https://huggingface.co/spaces/$SPACE_NAME $TEMP_DIR

# Copy backend files to the temp directory
echo "📋 Copying backend files..."
cp -r $BACKEND_DIR/* $TEMP_DIR/

# Rename Dockerfile.huggingface to Dockerfile
echo "🐳 Setting up Dockerfile..."
if [ -f "$TEMP_DIR/Dockerfile.huggingface" ]; then
  mv $TEMP_DIR/Dockerfile.huggingface $TEMP_DIR/Dockerfile
fi

# Go to the temp directory
cd $TEMP_DIR

# Add, commit, and push changes
echo "💾 Committing changes..."
git add .
git commit -m "Deploy backend to Hugging Face Spaces"

echo "📤 Pushing to Hugging Face Spaces..."
git push

echo "✅ Deployment complete! Your backend will be available at: https://huggingface.co/spaces/$SPACE_NAME"
echo "⏱️ Note: It may take a few minutes for the changes to be reflected."

# Clean up
echo "🧹 Cleaning up..."
cd -
rm -rf $TEMP_DIR

echo "Done! 🎉"