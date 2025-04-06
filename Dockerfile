# Use a slim Python base image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Ensure model files are copied to the right path (optional if already inside the project structure)
COPY models/heart_disease_model.pkl /app/models/
COPY models/scaler.pkl /app/models/

# Expose the FastAPI port
EXPOSE 8000

# Run the app using uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
