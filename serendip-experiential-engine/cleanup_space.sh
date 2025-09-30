#!/bin/bash
#
# cleanup_space.sh - Script to clean up storage on Hugging Face Spaces
# 
# This script helps manage the storage limitations on Hugging Face Spaces.
# It cleans up temporary files, model caches, and other storage-consuming
# artifacts to help prevent the 50GB storage limit error.
#

echo "Starting Hugging Face Space storage cleanup..."

# Create temp dirs if they don't exist
mkdir -p /tmp/transformers_cache /tmp/hf_home

# Set permissions
chmod 777 /tmp/transformers_cache /tmp/hf_home

# Find and remove large temporary files
echo "Removing large temporary files..."
find /tmp -type f -size +50M -exec rm -f {} \;

# Clean Hugging Face cache directories
echo "Cleaning HuggingFace cache directories..."
rm -rf /root/.cache/huggingface/*
rm -rf /home/*/.cache/huggingface/*

# Clean up Docker images that aren't being used
if command -v docker &> /dev/null; then
    echo "Cleaning up Docker resources..."
    docker system prune -f
fi

# Clean Python cache files
echo "Cleaning Python cache files..."
find / -type d -name "__pycache__" -exec rm -rf {} +  2>/dev/null || true
find / -name "*.pyc" -delete  2>/dev/null || true

# Remove log files
echo "Cleaning log files..."
find /var/log -type f -name "*.log" -exec rm -f {} \; 2>/dev/null || true
find /var/log -type f -name "*.log.*" -exec rm -f {} \; 2>/dev/null || true

# Show the current disk usage
echo "Current disk usage:"
df -h /

# Show largest directories
echo "Largest directories (top 10):"
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -10

echo "Cleanup completed."