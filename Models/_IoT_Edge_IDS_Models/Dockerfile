# Use Python 3.12 to perfectly match the Google Colab environment
FROM python:3.12-slim

# CRITICAL FIX: Force Python to print logs immediately without buffering
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install the EXACT library versions used during Colab training
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==2.0.2 \
    scikit-learn==1.6.1 \
    joblib==1.5.3 \
    pyarrow \
    tqdm

# Copy all the models, data, and the Python script into the container
COPY . /app

# Run the inference service when the container starts
CMD ["python", "inference_service.py"]