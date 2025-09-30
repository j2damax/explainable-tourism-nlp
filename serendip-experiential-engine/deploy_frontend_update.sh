#!/bin/bash

# deploy_frontend_update.sh
# Script to quickly deploy frontend changes to Hugging Face Spaces

echo "=========================================================="
echo "Deploying frontend updates to Hugging Face Spaces"
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
WORK_DIR=$(mktemp -d)
echo "Working in temporary directory: $WORK_DIR"
cd $WORK_DIR

# Test token validity first
echo "Testing Hugging Face API token validity..."
curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/spaces/$FRONTEND_SPACE > /tmp/hf_status
HF_STATUS=$(cat /tmp/hf_status)

if [ "$HF_STATUS" == "401" ] || [ "$HF_STATUS" == "403" ]; then
    echo "Error: Invalid Hugging Face token. Please check your HF_TOKEN."
    exit 1
fi

# Set up git configuration
echo "Configuring git..."
git init
git config --local user.email "you@example.com"
git config --local user.name "Your Name"

# Clone the space - using a different approach since we're already in a git directory
echo "Fetching content from existing space..."
# First set up the remote
git remote add hf "https://oauth2:$HF_TOKEN@huggingface.co/spaces/$FRONTEND_SPACE" || git remote set-url hf "https://oauth2:$HF_TOKEN@huggingface.co/spaces/$FRONTEND_SPACE"

# Try to fetch (will fail if repository doesn't exist yet, which is ok)
git fetch hf main || echo "Remote fetch failed - might be a new repository"

# Get current content if available (won't work for new repos, which is fine)
git checkout -b temp-branch || git checkout temp-branch
git reset --hard || true
git pull hf main || echo "No content to pull - might be a new repository"

# Copy updated files from local frontend directory
echo "Copying updated application files..."
cp -f "$LOCAL_FRONTEND_DIR/app.py" .

# Add all files to git
echo "Adding files to git..."
git add app.py

# Commit changes
git commit -m "Update visualization for Experience Dimension Analysis plot"

# Push to Hugging Face using the remote we set up
echo "Pushing to Hugging Face Spaces..."
git push -f hf HEAD:main

# Check the push result
if [ $? -ne 0 ]; then
    echo "Error: Failed to push to Hugging Face Spaces."
    echo "Let's try a different approach..."
    
    # Create a completely fresh attempt
    cd ..
    rm -rf "$WORK_DIR"
    WORK_DIR=$(mktemp -d)
    echo "Working in new temporary directory: $WORK_DIR"
    cd "$WORK_DIR"
    
    # Start fresh
    echo "Setting up a fresh repository..."
    git init
    git config --local user.email "you@example.com"
    git config --local user.name "Your Name"
    
    # Create necessary files
    echo "Creating Hugging Face Space configuration..."
    
    # Copy files from local frontend directory
    echo "Copying application files..."
    cp -r "$LOCAL_FRONTEND_DIR"/* .
    
    # Add all files to git
    git add .
    git commit -m "Update visualization for Experience Dimension Analysis plot"
    
    # Try direct push
    echo "Trying direct push to Hugging Face..."
    git push -f "https://oauth2:$HF_TOKEN@huggingface.co/spaces/$FRONTEND_SPACE" HEAD:main
    
    if [ $? -ne 0 ]; then
        echo "Error: All attempts to push to Hugging Face Spaces failed."
        exit 1
    fi
fi

# Clean up
cd - > /dev/null
rm -rf $WORK_DIR

echo "=========================================================="
echo "Update complete! Your application should now be available with improved visualization at:"
echo "https://huggingface.co/spaces/$FRONTEND_SPACE"
echo "=========================================================="