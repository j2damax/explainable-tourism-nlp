#!/bin/bash

# Master deployment script for Serendip Experiential Engine
# This script deploys both backend and frontend to Hugging Face Spaces

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set it with: export HF_TOKEN=your_hugging_face_token"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable not set"
    echo "GenAI Classification features will not work without an OpenAI API key"
    echo "You can set it with: export OPENAI_API_KEY=your_openai_api_key"
    
    # Check if it's in .env file
    if [ -f ".env" ] && grep -q "OPENAI_API_KEY" .env; then
        echo "Found OPENAI_API_KEY in .env file. Will use that for deployment."
    else
        echo "Consider adding OPENAI_API_KEY to your .env file for GenAI features"
        read -p "Do you want to continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Define the project root directory
PROJECT_ROOT="$(pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Deploy backend first
echo "==== DEPLOYING BACKEND ===="
echo "Changing to backend directory: $BACKEND_DIR"
cd "$BACKEND_DIR" || { echo "Error: Backend directory not found"; exit 1; }

# Check if deploy script exists, else create it
if [ ! -f "deploy_to_huggingface.sh" ]; then
    echo "Backend deployment script not found. Creating it..."
    cat > deploy_to_huggingface.sh << 'EOL'
#!/bin/bash

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
    git clone "https://user:$HF_TOKEN@huggingface.co/spaces/$SPACE_NAME" .
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

# Create README.md if it doesn't exist
if [ ! -f "README.md" ]; then
    echo "# Serendip Experiential Backend" > README.md
    echo "FastAPI backend for the Serendip Experiential Engine" >> README.md
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
cat > .hf/metadata.json << EOL2
{
    "app_port": 7860,
    "app_file": "api.py"
}
EOL2

# Add all files to git
git add .

# Commit changes
git commit -m "Deploy backend to Hugging Face Spaces"

# Push to Hugging Face Spaces
echo "Pushing to Hugging Face Spaces..."
git push -f "https://user:$HF_TOKEN@huggingface.co/spaces/$SPACE_NAME" main

# Clean up
cd - > /dev/null
rm -rf $TMP_DIR

echo "Deployment complete! Your backend should be available at:"
echo "https://huggingface.co/spaces/$SPACE_NAME"
EOL
    chmod +x deploy_to_huggingface.sh
fi

# Run backend deployment script
echo "Running backend deployment script..."
./deploy_to_huggingface.sh || { echo "Backend deployment failed"; exit 1; }

# Deploy frontend next
echo "==== DEPLOYING FRONTEND ===="
echo "Changing to frontend directory: $FRONTEND_DIR"
cd "$FRONTEND_DIR" || { echo "Error: Frontend directory not found"; exit 1; }

# Check if deploy script exists, else create it
if [ ! -f "deploy_to_huggingface.sh" ]; then
    echo "Frontend deployment script not found. Creating it..."
    cat > deploy_to_huggingface.sh << 'EOL'
#!/bin/bash

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set it with: export HF_TOKEN=your_hugging_face_token"
    exit 1
fi

# Define variables
SPACE_NAME="j2damax/serendip-experiential-frontend"
REPO_URL="https://huggingface.co/spaces/$SPACE_NAME"
LOCAL_DIR="$(pwd)"

echo "Deploying frontend to Hugging Face Spaces: $SPACE_NAME"

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
    git clone "https://user:$HF_TOKEN@huggingface.co/spaces/$SPACE_NAME" .
    # Remove all files except .git to ensure clean state
    find . -mindepth 1 -not -path "./.git*" -delete
else
    echo "Creating new space..."
    # Will push later to create the repository
fi

# Copy all files from the frontend directory
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

# Create README.md if it doesn't exist
if [ ! -f "README.md" ]; then
    echo "# Serendip Experiential Frontend" > README.md
    echo "Streamlit frontend for the Serendip Experiential Engine" >> README.md
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
cat > .hf/metadata.json << EOL2
{
    "app_port": 7860,
    "app_file": "app.py",
    "docker_build_args": {
        "API_URL": "https://j2damax-serendip-experiential-backend.hf.space"
    }
}
EOL2

# Add all files to git
git add .

# Commit changes
git commit -m "Deploy frontend to Hugging Face Spaces"

# Push to Hugging Face Spaces
echo "Pushing to Hugging Face Spaces..."
git push -f "https://user:$HF_TOKEN@huggingface.co/spaces/$SPACE_NAME" main

# Clean up
cd - > /dev/null
rm -rf $TMP_DIR

echo "Deployment complete! Your frontend should be available at:"
echo "https://huggingface.co/spaces/$SPACE_NAME"
EOL
    chmod +x deploy_to_huggingface.sh
fi

# Run frontend deployment script
echo "Running frontend deployment script..."
./deploy_to_huggingface.sh || { echo "Frontend deployment failed"; exit 1; }

# Return to the project root
cd "$PROJECT_ROOT"

echo "==== DEPLOYMENT COMPLETE ===="
echo "Backend: https://huggingface.co/spaces/j2damax/serendip-experiential-backend"
echo "Frontend: https://huggingface.co/spaces/j2damax/serendip-experiential-frontend"