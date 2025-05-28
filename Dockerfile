# Use Red Hat UBI 8 Python 3.9 image
FROM registry.access.redhat.com/ubi8/python-39:latest

# Set working directory
WORKDIR /opt/app-root/src

# Copy requirements.txt first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create embeddings directory in user's home
RUN mkdir -p /opt/app-root/embeddings

# Download and cache the embedding model
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/opt/app-root/embeddings'); \
    print('Model downloaded successfully')"


# Copy application code
COPY app.py ./

# Create directory for application data and set minimal permissions
RUN mkdir -p /opt/app-root/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/app-root/src
ENV HOME=/opt/app-root/src

# Expose port
EXPOSE 8080

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "app.py"]
