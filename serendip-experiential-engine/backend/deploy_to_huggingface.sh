#!/bin/bash

# Deployment script for Serendip Experiential Backend to Hugging Face Spaces

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set it with: export HF_TOKEN=your_hugging_face_token"
    exit 1
fi

# Define variables
SPACE_NAME="j2damax/serendip-experiential-backend"
REPO_URL="https://huggingface.co/spaces/$SPACE_NAME"
LOCAL_DIR="$(pwd)"

echo "Deploying backend to Hugging Face Spaces: $SPACE_NAME"

# Create a temporary directory
TMP_DIR=$(mktemp -d)
cd $TMP_DIR

# Initialize git and set credentials
git init
git config --local user.email "you@example.com"
git config --local user.name "Your Name"

# Clone the space if it exists, otherwise create from scratch
if curl --fail --silent -H "Authorization: Bearer $HF_TOKEN" $REPO_URL > /dev/null; then
    echo "Space exists, cloning repository..."
    # Format for Hugging Face API token authentication
    git clone "https://huggingface.co/spaces/$SPACE_NAME" .
    git config --local credential.helper store
    echo "https://oauth2:$HF_TOKEN@huggingface.co" > ~/.git-credentials
    # Remove all files except .git to ensure clean state
    find . -mindepth 1 -not -path "./.git*" -delete
else
    echo "Creating new space..."
    # Will push later to create the repository
fi

# Copy all files from the backend directory
echo "Copying files from $LOCAL_DIR to temporary directory..."
cp -r $LOCAL_DIR/* .

# Remove any unnecessary files
echo "Cleaning up unnecessary files..."
rm -rf __pycache__ .ipynb_checkpoints .pytest_cache .venv

# Use the Hugging Face specific Dockerfile
if [ -f "Dockerfile.huggingface" ]; then
    echo "Using Hugging Face specific Dockerfile..."
    mv Dockerfile.huggingface Dockerfile
fi

# Copy the README.md file with proper YAML metadata
if [ -f "$LOCAL_DIR/README.md" ]; then
    echo "Using existing README.md with YAML metadata..."
    cp "$LOCAL_DIR/README.md" ./README.md
else
    echo "# Creating default README.md with YAML metadata..."
    cat > README.md << EOL
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
FastAPI backend for the Serendip Experiential Engine
EOL
fi

# Create .gitignore
echo ".env
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.coverage
htmlcov/
.ipynb_checkpoints
*.ipynb
venv/
.venv/" > .gitignore

# Create Hugging Face Space metadata file
mkdir -p .hf
cat > .hf/metadata.json << EOL
{
    "app_port": 7860,
    "app_file": "api.py"
}
EOL

# Add all files to git
git add .

# Commit changes
git commit -m "Deploy backend to Hugging Face Spaces"

# Push to Hugging Face Spaces
echo "Pushing to Hugging Face Spaces..."
# Use stored credential helper instead of embedding in URL
git remote add origin "https://huggingface.co/spaces/$SPACE_NAME"
git push -f origin main

# Clean up
cd - > /dev/null
rm -rf $TMP_DIR

echo "Deployment complete! Your backend should be available at:"
echo "https://huggingface.co/spaces/$SPACE_NAME"