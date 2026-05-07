FROM python:3.11-slim

# Set working directory
WORKDIR /app


# Copy requirements first for better caching
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV HF_HOME=/app/models_cache

# Create cache directory for models
RUN mkdir -p /app/models_cache

# Run the application
CMD ["python", "main.py", "--run-now"]
