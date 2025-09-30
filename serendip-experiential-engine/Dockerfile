FROM python:3.10-slim

WORKDIR /app

# Install Docker and Docker Compose
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    docker.io \
    docker-compose \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Make the cleanup script executable
RUN chmod +x cleanup_space.sh

# Run the cleanup script before starting the application
RUN ./cleanup_space.sh

# Expose port for Hugging Face
EXPOSE 7860

# Create directories for HuggingFace transformers cache
RUN mkdir -p /tmp/transformers_cache /tmp/hf_home && chmod 777 /tmp/transformers_cache /tmp/hf_home

# Set environment variables
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache \
    HF_HOME=/tmp/hf_home

# Start Docker Compose when the container starts
# Hugging Face Spaces expects the app to be listening on port 7860
CMD ["docker-compose", "up"]