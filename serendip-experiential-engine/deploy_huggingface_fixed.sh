#!/bin/bash

# A simplified script to fix Hugging Face deployment issues
# This script focuses on proper authentication and error handling

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set it with: export HF_TOKEN=your_hugging_face_token"
    exit 1
fi

echo "=========================================================="
echo "Deploying to Hugging Face with improved authentication"
echo "=========================================================="

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

echo "Token appears valid. HTTP status: $HF_STATUS"

# Set up git configuration
echo "Configuring git..."
git init
git config --local user.email "you@example.com"
git config --local user.name "Your Name"

# Create necessary files for Hugging Face Space
echo "Creating Hugging Face Space configuration..."

# Create README with Space metadata
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

## Storage Optimization
This application uses limited caching to prevent hitting the 50GB storage limit on Hugging Face Spaces.
EOL

# Copy files from local frontend directory
echo "Copying application files..."
cp -r $LOCAL_FRONTEND_DIR/* .

# Create a cleanup script to help manage storage
cat > cleanup.sh << 'EOL'
#!/bin/bash
echo "Starting storage cleanup..."
find /tmp -type f -size +50M -delete 2>/dev/null || true
rm -rf ~/.cache/huggingface/* 2>/dev/null || true
find / -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
find / -name "*.pyc" -delete 2>/dev/null || true
df -h /
EOL
chmod +x cleanup.sh

# Set up HF metadata
mkdir -p .hf

# Get OpenAI API key if available
OPENAI_API_KEY_VALUE=""
if [ -n "$OPENAI_API_KEY" ]; then
    echo "Using OPENAI_API_KEY from environment variables"
    OPENAI_API_KEY_VALUE=$OPENAI_API_KEY
fi

# Create HF metadata
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
        "PIP_NO_CACHE_DIR": "1"
    }
}
EOL

# Add all files to git
echo "Adding files to git..."
git add .

# Commit changes
git commit -m "Deploy with storage optimization"

# Push to Hugging Face (use direct URL with token)
echo "Pushing to Hugging Face Spaces..."
git push -f "https://oauth2:$HF_TOKEN@huggingface.co/spaces/$FRONTEND_SPACE" HEAD:main

# Check the push result
if [ $? -ne 0 ]; then
    echo "Error: Failed to push to Hugging Face Spaces."
    echo "Please verify your token has write access to $FRONTEND_SPACE"
    exit 1
fi

# Clean up
cd - > /dev/null
rm -rf $WORK_DIR

echo "=========================================================="
echo "Deployment complete! Your application should now be available at:"
echo "https://huggingface.co/spaces/$FRONTEND_SPACE"
echo "=========================================================="