# Use Red Hat UBI 8 Python 3.9 image
FROM registry.access.redhat.com/ubi8/python-39:latest

# Set working directory
WORKDIR /opt/app-root/src

# Copy requirements.txt first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py ./

# Create directories for mounted volumes and ensure proper permissions
RUN mkdir -p /mnt/embeddings /mnt/milvus /opt/app-root/data && \
    chmod -R g+rwX /mnt/embeddings /mnt/milvus /opt/app-root/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/app-root/src
ENV HOME=/opt/app-root/src
ENV EMBEDDING_MODEL_PATH=/mnt/embeddings
ENV MILVUS_DATA_PATH=/mnt/milvus

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "app.py"]
