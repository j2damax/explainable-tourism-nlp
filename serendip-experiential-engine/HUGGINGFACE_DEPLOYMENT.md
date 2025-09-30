# Deployment to Hugging Face Spaces

This guide provides instructions for deploying the Serendip Travel Experience Engine to Hugging Face Spaces while optimizing for the 50GB storage limit.

## Storage Optimization

The Hugging Face Spaces deployment has been optimized to avoid storage limit errors:

1. **Temporary Model Storage**: Models are loaded to temporary directories (`/tmp`) to prevent accumulating model files in persistent storage
2. **Cache Cleanup**: Periodic cleanup of unused temporary files
3. **Reduced Image Quality**: Visualization images are generated with lower resolution to save space
4. **Memory Optimizations**: Model loading uses `low_cpu_mem_usage=True` to reduce memory footprint
5. **Container Resources**: Memory limits are applied to containers in `docker-compose.yml`
6. **No Volumes**: Production deployment removes volumes to prevent persistent storage accumulation

## Deployment Steps

1. **Create a new Space on Hugging Face**:

   - Go to https://huggingface.co/spaces
   - Click "New Space"
   - Select "Docker" as the Space SDK
   - Configure with at least 16GB of RAM

2. **Clone your Space repository**:

   ```bash
   git clone https://huggingface.co/spaces/your-username/your-space-name
   cd your-space-name
   ```

3. **Copy the application files**:

   ```bash
   # Copy backend, frontend, and docker-compose.yml
   cp -r /path/to/serendip-experiential-engine/* .
   ```

4. **Add a Dockerfile for Hugging Face**:
   Create a `Dockerfile` in the root directory:

   ```dockerfile
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

   # Copy the application
   COPY . .

   # Expose ports
   EXPOSE 7860

   # Start Docker Compose when the container starts
   CMD ["docker-compose", "up"]
   ```

5. **Create .dockerignore file**:

   ```
   # Ignore large files and directories
   __pycache__/
   *.py[cod]
   *$py.class
   *.so
   .env
   .venv
   venv/
   ENV/
   ```

6. **Commit and push to Hugging Face**:
   ```bash
   git add .
   git commit -m "Deploy Serendip Travel Engine with storage optimizations"
   git push
   ```

7. **Monitor the deployment**:
   - Check your Space's build logs for errors
   - Monitor storage usage regularly with the commands below:
   ```bash
   # Check disk usage
   df -h /
   
   # Monitor Docker containers
   docker ps -a
   
   # View application logs
   docker-compose logs -f
   
   # Monitor space usage growth over time
   watch -n 60 "df -h / && echo '---------' && du -sh /tmp/* 2>/dev/null"
   ```

## Using the Cache Reset and Redeployment Script

The repository includes a `deploy_reset_cache.sh` script that automates the process of cleaning up storage and redeploying the application:

1. **SSH into your Hugging Face Space**:
   
   - Go to your Space settings
   - Find the SSH option and connect to your Space

2. **Run the deployment script**:

   ```bash
   cd /app
   chmod +x deploy_reset_cache.sh
   ./deploy_reset_cache.sh
   ```

The script will:
- Clean up temporary files, model caches, and Docker resources
- Stop running containers
- Rebuild containers without using cache
- Set environment variables for storage optimization
- Restart the application with optimized settings

## Troubleshooting Storage Issues

If you still encounter storage limit issues after running the deployment script:

1. **Check the storage usage**:

   - SSH into the Space from the HF interface
   - Run `du -h --max-depth=1 /` to see which directories use the most space
   - Look for unexpected large directories: `find / -type d -exec du -sh {} \; | sort -hr | head -n 20`

2. **Manual cleanup**:

   ```bash
   # Clean model cache files
   find /tmp -type f -name "*.safetensors" -delete
   find /tmp -type f -name "*.bin" -delete
   find / -path "*/huggingface/hub/*" -type f -size +10M -delete
   
   # Clean Docker resources
   docker system prune -af --volumes
   
   # Remove Python cache
   find / -name "__pycache__" -type d -exec rm -rf {} +
   find / -name "*.pyc" -delete
   ```

3. **Use a smaller model**:

   - Consider using a more compact model variant
   - Change the `MODEL_NAME` in `backend/api.py` to a smaller model like "distilbert-base-uncased"
   - Modify the model loading with quantization parameters:
     ```python
     model = AutoModelForSequenceClassification.from_pretrained(
         MODEL_NAME, 
         num_labels=NUM_LABELS,
         problem_type="multi_label_classification",
         low_cpu_mem_usage=True,
         load_in_8bit=True  # Enable quantization to reduce memory usage
     )
     ```

4. **Disable explainability feature**:
   - The visualization generation consumes significant resources
   - Consider disabling or simplifying the `/explain` endpoint
   - Reduce the resolution of generated visualizations further
