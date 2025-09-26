.PHONY: build up down test clean

# Build the Docker containers
build:
	docker compose build

# Start the services
up:
	docker compose up

# Start the services in detached mode
up-d:
	docker compose up -d

# Stop the services
down:
	docker compose down

# Run the tests
test:
	cd tests && bash run_tests.sh

# Clean up Docker resources
clean:
	docker compose down --rmi local -v
	docker system prune -f

# Run the backend service locally for development
backend-dev:
	cd backend && uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Run the frontend service locally for development
frontend-dev:
	cd frontend && streamlit run app.py