#!/bin/bash

# deploy_reset_cache.sh
# Script to reset cache and redeploy the Serendip Travel Experience Engine on Hugging Face Spaces

echo "========================================================"
echo "Redeploying Serendip Travel Experience Engine with cache reset"
echo "========================================================"

# Run the cleanup script first
echo "Step 1: Cleaning up storage and cache..."
./cleanup_space.sh

# Make sure the Docker daemon is running
echo "Step 2: Ensuring Docker daemon is running..."
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker daemon is not running. Please start Docker first."
    exit 1
fi

# Stop any running containers
echo "Step 3: Stopping existing containers..."
docker-compose down

# Rebuild the containers without using cache
echo "Step 4: Rebuilding containers without cache..."
docker-compose build --no-cache

# Set environment variables for minimal storage usage
echo "Step 5: Setting environment variables for storage optimization..."
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/hf_home
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Add these environment variables to the .env file while preserving other variables
echo "Step 6: Updating .env file with storage optimization while preserving other variables..."
if [ -f .env ]; then
    # Create a backup of the existing .env file
    cp .env .env.backup
    
    # Extract the OpenAI API key from the existing .env file
    OPENAI_API_KEY=$(grep "OPENAI_API_KEY" .env.backup | cut -d '=' -f2-)
    OPENAI_MODEL=$(grep "OPENAI_MODEL" .env.backup | cut -d '=' -f2-)
    
    # If no OpenAI API key found in the current .env, use a default value
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "No OpenAI API key found in current .env, checking for key in tourism-review-classification/.env"
        if [ -f ../tourism-review-classification/.env ]; then
            OPENAI_API_KEY=$(grep "OPENAI_API_KEY" ../tourism-review-classification/.env | cut -d '=' -f2-)
            echo "Using API key from tourism-review-classification/.env"
        fi
    fi
    
    # If OPENAI_MODEL is not found, use a default value
    if [ -z "$OPENAI_MODEL" ]; then
        OPENAI_MODEL="gpt-3.5-turbo"
    fi
fi

# Create the new .env file with both storage optimization and existing variables
cat > .env << EOF
# Environment variables for storage optimization
TRANSFORMERS_CACHE=/tmp/transformers_cache
HF_HOME=/tmp/hf_home
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
PIP_NO_CACHE_DIR=1
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Backend environment variables
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=info

# Frontend environment variables
API_URL=http://backend:8000

# OpenAI API key for GenAI benchmark comparisons
OPENAI_API_KEY=${OPENAI_API_KEY}
OPENAI_MODEL=${OPENAI_MODEL}
EOF

# Start the application
echo "Step 7: Starting the application..."
docker-compose up -d

# Check the running containers
echo "Step 8: Verifying running containers..."
docker-compose ps

echo "========================================================"
echo "Deployment complete! The application should now be running."
echo "If you still encounter storage issues, check the logs with:"
echo "docker-compose logs -f"
echo "========================================================"

# Check storage status after deployment
echo "Storage status:"
df -h /

echo "Done."