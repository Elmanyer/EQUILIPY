#!/bin/bash

# Usage: ./check_and_fix_separators.sh /path/to/directory

# Check if correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Get the target directory from the argument
TARGET_DIR="$1"

# Loop through all directories in the target directory
for folder in "$TARGET_DIR"/*/; do
    # Get the base name of the folder (without the full path)
    folder_name=$(basename "$folder")

    # Construct the path to the expected .dat file
    dat_file="$folder/$folder_name.dat"

    # Check if the .dat file exists
    if [ -f "$dat_file" ]; then
        # Check if the file contains the incorrect separator (hB---------------)
        if grep -q 'hB------------------------------------------------------------------' "$dat_file"; then
            # Replace all instances of hB--------------- with $---------------------------------------------
            sed -i 's/hB------------------------------------------------------------------/$-------------------------------------------------------------------/g' "$dat_file"
            echo "Corrected separators in file: $dat_file"
        else
            echo "No incorrect separators found in file: $dat_file"
        fi
    else
        echo "No .dat file found in folder: $folder_name"
    fi
done

echo "Separator check and correction completed."
