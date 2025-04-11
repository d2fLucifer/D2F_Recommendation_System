#!/bin/bash

# Create the html directory and subdirectories for ACME challenges
mkdir -p ./html/.well-known/acme-challenge

# Set permissions to ensure NGINX can read the files
chmod -R 755 ./html

# Create a test file to verify the challenge path
echo "test" > ./html/.well-known/acme-challenge/test-file

echo "Setup complete. Test the challenge path with: curl http://d2f.io.vn/.well-known/acme-challenge/test-file"