# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Default command (can be changed to app.py if desired)
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]

