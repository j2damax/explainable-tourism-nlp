#!/bin/bash
# monitor_storage.sh
# Script to monitor storage usage on Hugging Face Spaces

echo "==============================================="
echo "Storage Monitoring for Hugging Face Spaces"
echo "Press Ctrl+C to exit"
echo "==============================================="

# Function to display storage information
show_storage_info() {
    clear
    echo "TIMESTAMP: $(date)"
    echo "==============================================="
    echo "DISK USAGE SUMMARY:"
    df -h / | grep -v "Filesystem"
    
    echo "==============================================="
    echo "TOP 10 LARGEST DIRECTORIES:"
    du -sh /* 2>/dev/null | sort -hr | head -10
    
    echo "==============================================="
    echo "HUGGING FACE CACHE SIZE:"
    du -sh /tmp/transformers_cache 2>/dev/null || echo "No HF cache found in /tmp"
    du -sh /tmp/hf_home 2>/dev/null || echo "No HF home found in /tmp"
    du -sh ~/.cache/huggingface 2>/dev/null || echo "No HF cache found in home dir"
    
    echo "==============================================="
    echo "DOCKER RESOURCES:"
    docker ps -a --size 2>/dev/null || echo "Docker not available"
    
    echo "==============================================="
    echo "LARGEST FILES IN /tmp:"
    find /tmp -type f -size +10M 2>/dev/null | xargs du -sh 2>/dev/null | sort -hr | head -5
    
    echo "==============================================="
    echo "MEMORY USAGE:"
    free -h
    
    echo "==============================================="
    echo "TOP PROCESSES BY MEMORY:"
    ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -10
    
    echo "==============================================="
}

# Monitor in a loop
while true; do
    show_storage_info
    echo "Refreshing in 30 seconds... (Press Ctrl+C to exit)"
    sleep 30
done