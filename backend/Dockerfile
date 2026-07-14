FROM python:3.10-slim

# Install system dependencies for OpenCV and styling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpng-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI port
EXPOSE 8000

ENV UPLOAD_DIR=/workspace/data/uploads
ENV REPORTS_DIR=/workspace/data/reports
ENV MODEL_DIR=/workspace/data/models
ENV DATABASE_URL=sqlite:////workspace/data/microscopy.db

VOLUME /workspace/data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
