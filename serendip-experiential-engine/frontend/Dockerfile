FROM python:3.10-slim

WORKDIR /app

# Set environment variables to reduce container size
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir python-dotenv

# Copy application code
COPY . .

EXPOSE 8501

# Set API URL environment variable
ENV API_URL=http://backend:8000

# Run Streamlit with optimized settings
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]