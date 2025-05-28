# Use Red Hat UBI 8 with Python 3.9
FROM registry.access.redhat.com/ubi8/python-39:latest

# Set working directory
WORKDIR /app

# Create a non-root user and group
RUN groupadd -g 1001 appuser && \
    useradd -u 1001 -g appuser -m -d /home/appuser appuser

# Install system dependencies
RUN dnf install -y gcc python3-devel && \
    dnf clean all

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model and store it in /embeddings
RUN python -c "from transformers import AutoTokenizer; from langchain_huggingface import HuggingFaceEmbeddings; \
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_folder='/embeddings')"

# Copy the application code
COPY app.py .

# Set permissions for /app and /embeddings
RUN chown -R appuser:appuser /app /embeddings && \
    chmod -R u+rw /app /embeddings

# Create directory for Milvus data with appropriate permissions
RUN mkdir -p /mnt/milvus && \
    chown -R appuser:appuser /mnt/milvus && \
    chmod -R u+rw /mnt/milvus

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
