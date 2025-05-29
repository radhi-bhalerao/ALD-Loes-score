#!/bin/bash

# File containing the list of lines
input_file="/data/rauschecker1/lesionSegmentation/ALD_cont/radhika/data/air_download/ald_accessions.txt"

# Path to config.ini (update if necessary)
config_path="/data/rauschecker1/lesionSegmentation/ALD_cont/pyalfe/config_ald.ini"

# Check if the file exists
if [[ ! -f "$input_file" ]]; then
    echo "Error: $input_file does not exist."
    exit 1
fi

# Loop through each line in the file
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines
    if [[ -z "$line" ]]; then
        continue
    fi
    
    # Run the pyalfe command with the line replacing P0007
    echo "Running: pyalfe run -c $config_path $line"
    pyalfe run -c "$config_path" "$line"
done < "$input_file"
