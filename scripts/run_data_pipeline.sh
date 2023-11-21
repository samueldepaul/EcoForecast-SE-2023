#!/bin/bash

# You can run this script from the command line using:
# ./run_data_pipeline.sh <start_date> <end_date> <raw_data_file> <processed_data_file> <model_file> <predictions_file>
# For example:
# ./run_data_pipeline.sh 2022-01-01 2023-01-01 data/ data/ models/ predictions/

# Get command line arguments
start_date="$1"
end_date="$2"
raw_data_file="$3"
processed_data_file="$4"
model_file="$5"
predictions_file="$6"

# Run data_ingestion.py
echo " "
echo "Starting data ingestion..."
python3 src/data_ingestion.py --start_time "$start_date" --end_time "$end_date" --output_path "$raw_data_file"

# Run data_processing.py
echo " "
echo "Starting data processing..."
python3 src/data_processing.py --file_path="$raw_data_file" --output_path="$processed_data_file"

# Run model_training.py
echo " "
echo "Starting model training..."
python3 src/model_training.py --input_file="$processed_data_file" --model_file="$model_file"

# Run model_prediction.py
echo " "
echo "Starting prediction..."
python3 src/model_prediction.py --input_file="$processed_data_file" --model_file="$model_file" --output_file="$predictions_file"

echo " "
echo "Pipeline completed."
