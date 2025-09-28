#!/bin/bash

# Simple setup script for the Serendip Experiential Engine

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}======================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}======================================================${NC}\n"
}

# Check if Docker is installed
print_header "Checking prerequisites"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker Desktop first.${NC}"
    exit 1
else
    echo -e "${GREEN}✓${NC} Docker is installed"
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
else
    echo -e "${GREEN}✓${NC} Docker Compose is installed"
fi

# Check if services are already running
print_header "Checking for running services"
if docker ps | grep -q "serendip-experiential-engine"; then
    echo -e "${YELLOW}Serendip services are already running. Would you like to restart them? (y/n)${NC}"
    read -r restart
    if [[ $restart =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Stopping existing services...${NC}"
        docker-compose down
    else
        echo -e "${GREEN}Using existing running services.${NC}"
    fi
fi

# Start services
print_header "Starting Serendip Experiential Engine services"
echo -e "${YELLOW}This may take a few minutes on first run as it downloads images and builds containers...${NC}"
docker-compose up -d

# Check if services started successfully
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to start services. Check the logs with 'docker-compose logs'.${NC}"
    exit 1
fi

# Wait for services to be ready
print_header "Waiting for services to start"
echo -e "${YELLOW}Giving the FastAPI backend time to load the model...${NC}"
sleep 10

# Test backend health
print_header "Testing API health"
echo -e "${YELLOW}Checking if the backend API is healthy...${NC}"
health_response=$(curl -s http://localhost:8000/health)

if echo "$health_response" | grep -q "status.*ok"; then
    echo -e "${GREEN}✓${NC} Backend API is healthy and ready!"
else
    echo -e "${RED}Backend API is not responding correctly. Response: $health_response${NC}"
    echo -e "${YELLOW}You may need to wait longer for the model to load. Check logs with 'docker-compose logs backend'${NC}"
fi

# Print URLs for services
print_header "Access Information"
echo -e "${GREEN}Serendip Experiential Engine is now running!${NC}"
echo -e "\n${YELLOW}Services:${NC}"
echo -e "  - Backend API: ${GREEN}http://localhost:8000${NC}"
echo -e "  - Frontend UI: ${GREEN}http://localhost:8501${NC}"

echo -e "\n${YELLOW}Documentation:${NC}"
echo -e "  - API docs: ${GREEN}http://localhost:8000/docs${NC}"

echo -e "\n${BLUE}======================================================${NC}"
echo -e "${GREEN}Setup complete! Enjoy your Serendip Experiential Engine!${NC}"
echo -e "${BLUE}======================================================${NC}\n"