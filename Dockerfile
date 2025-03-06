# Use the official Apache Airflow image as the base
FROM apache/airflow:2.7.1-python3.9

# Switch to the root user to install system packages
USER root

# Update package lists and install necessary packages including curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        openjdk-11-jdk \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure Spark JAR directory exists and download the Qdrant Spark connector JAR file
RUN mkdir -p /usr/local/airflow/spark/jars && \
    curl -fSL https://repo1.maven.org/maven2/io/qdrant/spark/2.3.2/spark-2.3.2.jar -o /usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Switch back to the airflow user
USER airflow

# Copy the requirements file into the image
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Note:
# Removed the following lines because initializing the database should be done at runtime,
# not during the build process.
RUN airflow db init && \
    airflow db upgrade

# Set default entrypoint command as per the base image's expectations
CMD ["webserver"]