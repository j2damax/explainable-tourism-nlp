#!/bin/bash

# force_update_deploy.sh
# A script to force a clear update of the frontend with clear cache directives

echo "=========================================================="
echo "Forcing a clean update of frontend to Hugging Face Spaces"
echo "=========================================================="

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set it with: export HF_TOKEN=your_hugging_face_token"
    exit 1
fi

# Define spaces and directories
FRONTEND_SPACE="j2damax/serendip-experiential-frontend"
LOCAL_FRONTEND_DIR="/Users/jam/serendip-travel/explainable-tourism-nlp/serendip-experiential-engine/frontend"

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
echo "Working in temporary directory: $TEMP_DIR"
cd "$TEMP_DIR"

echo "Creating a new repository..."
git init
git config --local user.email "huggingface@example.com"
git config --local user.name "HuggingFace Deployment"

# Create README with Space metadata and cache buster
TIMESTAMP=$(date +%s)
echo "Creating Space configuration files with cache buster: $TIMESTAMP"
cat > README.md << EOL
---
title: Serendip Experiential Frontend
emoji: ✨
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# Serendip Experiential Frontend
Streamlit frontend for the Serendip Experiential Engine
Updated: $TIMESTAMP
EOL

# Copy all frontend files
echo "Copying application files..."
cp -r "$LOCAL_FRONTEND_DIR"/* .

# Create a basic .gitignore
cat > .gitignore << EOL
__pycache__/
*.py[cod]
*$py.class
.env
.venv/
venv/
EOL

# Create HF metadata directory
mkdir -p .hf

# Get OpenAI API key if available
OPENAI_API_KEY_VALUE=""
if [ -n "$OPENAI_API_KEY" ]; then
    echo "Using OPENAI_API_KEY from environment variables"
    OPENAI_API_KEY_VALUE=$OPENAI_API_KEY
fi

# Create HF metadata with cache control
cat > .hf/metadata.json << EOL
{
    "app_port": 7860,
    "app_file": "app.py",
    "docker_build_args": {
        "API_URL": "https://j2damax-serendip-experiential-backend.hf.space",
        "TRANSFORMERS_CACHE": "/tmp/transformers_cache",
        "HF_HOME": "/tmp/hf_home",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PIP_NO_CACHE_DIR": "1",
        "CACHE_BUSTER": "$TIMESTAMP"
    }
}
EOL

# Add all files to git
echo "Adding files to git..."
git add .

# Commit changes with explicit message
git commit -m "CRITICAL UPDATE: Remove warnings and fix Experience Dimension Analysis plot display (timestamp: $TIMESTAMP)"

# Push to Hugging Face with force
echo "Pushing to Hugging Face Spaces with force..."
git push -f "https://oauth2:$HF_TOKEN@huggingface.co/spaces/$FRONTEND_SPACE" HEAD:main

# Check the push result
if [ $? -ne 0 ]; then
    echo "Error: Failed to push to Hugging Face Spaces."
    echo "Please check your token and make sure you have access to the space."
    exit 1
else
    echo "Deployment successful!"
fi

# Clean up
cd - > /dev/null
rm -rf "$TEMP_DIR"

echo "=========================================================="
echo "Update complete! Your application should now be available with fixed visualization at:"
echo "https://huggingface.co/spaces/$FRONTEND_SPACE"
echo ""
echo "IMPORTANT: You may need to hard-refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)"
echo "to clear the cache and see the updated version."
echo "=========================================================="
