FROM python:3.9

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y netcat-traditional dos2unix curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /app/kafka

# Copy and install Python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Convert entrypoint script line endings and make it executable
RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set entrypoint
ENTRYPOINT ["/bin/bash", "/app/kafka/entrypoint.sh"]
