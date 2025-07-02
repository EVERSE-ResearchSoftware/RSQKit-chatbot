#!/bin/bash
# Script: download_rsqkit_files.sh
# Purpose: Download RSQKit files using scrapper.py
# Requirements:
# - Virtual environment (.venv) must exist and be activated
# - The directory './rsqkit_markdown' will be created if it does not exist

# Exit immediately if any command fails
set -e

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Ensure the required directory exists (create it if it doesn't)
echo "Ensuring the directory './rsqkit_markdown' exists..."
mkdir -p ./rsqkit_markdown

# Inform user about the process
echo "Downloading RSQKit files..."
echo "This can take a moment..."

# Execute the Python script to download files
python3 rsqkit_scrap.py

# Indicate completion
echo "Done."