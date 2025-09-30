#!/bin/bash

# deploy_huggingface_reset_storage.sh
# Script to reset the storage on Hugging Face Spaces and redeploy
# This addresses the "Workload evicted, storage limit exceeded (50G)" error

echo "========================================================"
echo "Deploying to Hugging Face with storage cleanup for Serendip Experiential Engine"
echo "========================================================"

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set it with: export HF_TOKEN=your_hugging_face_token"
    exit 1
fi

# Create cleanup command script for frontend
echo "Creating Hugging Face cleanup script..."

CLEANUP_SCRIPT=$(cat <<'EOL'
#!/bin/bash
# Cleanup script to run on Hugging Face Spaces
echo "Starting storage cleanup..."

# Remove cache directories
rm -rf /tmp/* || true
rm -rf ~/.cache/* || true
rm -rf ~/.huggingface/* || true
rm -rf /root/.cache/* || true

# Remove Python cache
find / -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
find / -name "*.pyc" -delete 2>/dev/null || true

# Remove log files
find /var/log -name "*.log*" -delete 2>/dev/null || true

# Clean Docker
docker system prune -af || true

# Show disk usage
df -h /
EOL
)

# Define spaces
FRONTEND_SPACE="j2damax/serendip-experiential-frontend"
BACKEND_SPACE="j2damax/serendip-experiential-backend"

# Deploy frontend with cleanup
echo "Deploying frontend with storage cleanup..."
cd frontend

# Create a temporary directory for frontend deployment
FRONTEND_TMP_DIR=$(mktemp -d)
cd $FRONTEND_TMP_DIR

# Initialize git and set git credentials globally
git init
git config --local user.email "you@example.com" 
git config --local user.name "Your Name"

# Set git credentials for Hugging Face
echo "Setting up git credentials for Hugging Face..."
git config --global credential.helper store
echo "https://oauth2:$HF_TOKEN@huggingface.co" > ~/.git-credentials
chmod 600 ~/.git-credentials

# Clone the existing space
echo "Cloning existing frontend space..."
git clone "https://oauth2:$HF_TOKEN@huggingface.co/spaces/$FRONTEND_SPACE" .
if [ $? -ne 0 ]; then
    echo "Cloning failed. Creating new repository structure..."
    # If clone fails, create the basic structure without cloning
    mkdir -p .git
    echo "ref: refs/heads/main" > .git/HEAD
fi

# Create cleanup script on Hugging Face
echo "$CLEANUP_SCRIPT" > cleanup.sh
chmod +x cleanup.sh

# Create custom README with storage optimization instructions
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
This application has been optimized to reduce storage usage on Hugging Face Spaces:
1. Limited caching of analyses
2. Storage cleanup on startup
3. Environment variables to minimize disk usage

If you encounter "Workload evicted, storage limit exceeded (50G)" errors, run the included cleanup.sh script.
EOL

# Copy latest app files from the local frontend directory
cp -r /Users/jam/serendip-travel/explainable-tourism-nlp/serendip-experiential-engine/frontend/* .

# Create startup script to clean storage on each start
cat > startup.sh << EOL
#!/bin/bash
# Run cleanup on startup
echo "Running storage cleanup on startup..."
bash ./cleanup.sh

# Start the streamlit app
streamlit run app.py
EOL
chmod +x startup.sh

# Update the Dockerfile to use our startup script
cat > Dockerfile << EOL
FROM python:3.9-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for storage optimization
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_home
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Copy the rest of the application
COPY . .

# Make scripts executable
RUN chmod +x startup.sh cleanup.sh

# Run the startup script which includes cleanup
CMD ["./startup.sh"]
EOL

# Create a .gitignore file
cat > .gitignore << EOL
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.coverage
htmlcov/
.ipynb_checkpoints
venv/
.venv/
EOL

# Set up HF metadata
mkdir -p .hf

# Get OpenAI API key
OPENAI_API_KEY_VALUE=""
if [ -n "$OPENAI_API_KEY" ]; then
    echo "Using OPENAI_API_KEY from environment variables"
    OPENAI_API_KEY_VALUE=$OPENAI_API_KEY
elif [ -f "/Users/jam/serendip-travel/explainable-tourism-nlp/serendip-experiential-engine/.env" ] && grep -q "OPENAI_API_KEY" "/Users/jam/serendip-travel/explainable-tourism-nlp/serendip-experiential-engine/.env"; then
    echo "Using OPENAI_API_KEY from .env file"
    OPENAI_API_KEY_VALUE=$(grep "OPENAI_API_KEY" "/Users/jam/serendip-travel/explainable-tourism-nlp/serendip-experiential-engine/.env" | cut -d '=' -f2-)
fi

# Create HF metadata
cat > .hf/metadata.json << EOL
{
    "app_port": 7860,
    "app_file": "app.py",
    "docker_build_args": {
        "API_URL": "https://j2damax-serendip-experiential-backend.hf.space",
        "OPENAI_API_KEY": "$OPENAI_API_KEY_VALUE",
        "TRANSFORMERS_CACHE": "/tmp/transformers_cache",
        "HF_HOME": "/tmp/hf_home",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PIP_NO_CACHE_DIR": "1"
    }
}
EOL

# Commit all changes
git add .
git commit -m "Deploy with storage optimization and cleanup"

# Push to Hugging Face with explicit URL instead of relying on origin
echo "Pushing optimized frontend to Hugging Face..."
git push -f "https://oauth2:$HF_TOKEN@huggingface.co/spaces/$FRONTEND_SPACE" main

# Clean up
cd /Users/jam/serendip-travel/explainable-tourism-nlp/serendip-experiential-engine
rm -rf $FRONTEND_TMP_DIR

echo "========================================================"
echo "Deployment complete! Your application should now run without hitting storage limits."
echo "Deployed to: https://huggingface.co/spaces/$FRONTEND_SPACE"
echo "========================================================"