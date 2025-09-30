# Complete Reset and Redeployment Guide

This guide provides step-by-step instructions for completely resetting and redeploying the Serendip Travel Experience Engine on Hugging Face Spaces when encountering persistent storage issues.

## Complete Reset Process

Follow these steps if you need to completely reset your Hugging Face Space:

### 1. Factory Reboot (First Attempt)

1. Go to your Hugging Face Space settings
2. Click on "Factory reboot"
3. Wait for the Space to rebuild and restart

If this doesn't resolve the issue, proceed to the manual reset steps below.

### 2. Manual Reset via SSH

1. SSH into your Space using the terminal option in Hugging Face
2. Execute these commands to clean up the Space:

```bash
# Stop all running containers
docker-compose down
docker stop $(docker ps -a -q) 2>/dev/null

# Remove all containers
docker rm $(docker ps -a -q) 2>/dev/null

# Remove all Docker images
docker rmi $(docker images -q) -f 2>/dev/null

# Clean Docker system
docker system prune -af --volumes

# Remove cache directories
rm -rf /tmp/*
rm -rf ~/.cache/huggingface
rm -rf /root/.cache/huggingface

# Clean up any large files
find / -type f -size +100M -not -path "/proc/*" -not -path "/sys/*" -exec rm -f {} \; 2>/dev/null
```

### 3. Rebuild from Source

After cleaning, redeploy the application:

```bash
# Clone your repository or copy files if already present
# Navigate to the application directory
cd /app

# Make scripts executable
chmod +x deploy_reset_cache.sh cleanup_space.sh monitor_storage.sh

# Run the deployment script with cache reset
./deploy_reset_cache.sh
```

### 4. Verify Deployment

Check that everything is working correctly:

```bash
# Monitor the deployment
./monitor_storage.sh

# Check logs
docker-compose logs -f
```

## Optimizing for Hugging Face Spaces

### Storage-Saving Tips

1. **Use temporary storage**:
   - Set `TRANSFORMERS_CACHE` and `HF_HOME` environment variables to `/tmp`
   - This prevents model files from filling persistent storage

2. **Reduce model size**:
   - Consider using a smaller model variant
   - Use quantization with `load_in_8bit=True`
   - Disable unnecessary model features

3. **Limit visualization quality**:
   - Use lower DPI settings for matplotlib (80 instead of 300)
   - Reduce image sizes and quality
   - Limit the number of visualizations stored

4. **Implement regular cleanup**:
   - Add a cron job for periodic cleanup
   - Set up automatic file pruning based on age or size

### Monitoring Best Practices

1. Set up regular monitoring with `./monitor_storage.sh`
2. Watch for directories that grow rapidly
3. Check Docker container sizes regularly
4. Monitor memory usage alongside storage

## Advanced Troubleshooting

If you still encounter issues, consider:

1. **Contacting Hugging Face Support**:
   - Share your Space URL and error details
   - Ask about increasing storage limits

2. **External Model Hosting**:
   - Host your model on a separate service
   - Configure the application to download the model on demand

3. **Split Your Application**:
   - Deploy frontend and backend in separate Spaces
   - Use API calls between them