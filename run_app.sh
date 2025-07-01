#!/bin/bash

echo "Activating virtual environment"
source .venv/bin/activate

echo "Starting the app"
streamlit run app.py


