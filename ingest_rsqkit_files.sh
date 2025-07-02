#!/bin/bash
# Script: ingest_rsqkit_files.sh
# Purpose: Ingest markdown data into Chroma using Ollama provider
# Requirements:
# - Virtual environment (.venv) must exist and be activated

# Set script to exit on error
set -e

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Ensure the required directory exists (create it if it doesn't)
echo "Ensuring the directory './rsqkit_markdown' exists..."
mkdir -p ./rsqkit_markdown

# Inform user about the process
echo "Starting data ingestion process. This may take several minutes..."

# Execute the Python script with specified parameters
python3 chroma_data_ingestor.py \
    --input_directory ./rsqkit_markdown \
    --provider Ollama \
    --collection rsqkit

# If we reach this point, the process completed successfully
echo "Data ingestion process completed successfully."