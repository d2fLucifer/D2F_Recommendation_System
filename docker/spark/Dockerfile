# Use the Bitnami Spark base image
FROM bitnami/spark:3.4.0

# Set environment variables
ENV SPARK_HOME=/opt/bitnami/spark

# Switch to root to install packages
USER root

# Install curl and clean up
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# (Optional) Install additional packages or dependencies
# For example, installing Python packages
# RUN pip install --no-cache-dir pandas numpy

# (Optional) Copy custom Spark configurations
# You can add your custom spark-defaults.conf or other config files
# COPY config/spark-defaults.conf $SPARK_HOME/conf/
# COPY config/spark-env.sh $SPARK_HOME/conf/

# Download and add Qdrant Spark connector
RUN curl -fSL https://repo1.maven.org/maven2/io/qdrant/spark/2.3.2/spark-2.3.2.jar -o qdrant-spark-2.3.2.jar && \
    mkdir -p $SPARK_HOME/jars/ && \
    mv qdrant-spark-2.3.2.jar $SPARK_HOME/jars/ 

# Revert to non-root user (adjust UID if necessary)
USER 1001

# Copy Spark jobs to the image
COPY ./jobs /opt/bitnami/spark/jobs

# Copy any additional data if needed
COPY ./data /opt/bitnami/spark/data

# (Optional) Set the entrypoint if you need to customize it
# ENTRYPOINT ["/opt/bitnami/scripts/spark/entrypoint.sh"]

# (Optional) Expose necessary ports (since ports are handled in docker-compose)
# EXPOSE 8080 7077

# Set the default command (can be overridden in docker-compose)
# CMD ["spark-class", "org.apache.spark.deploy.master.Master"]
