#!/bin/bash

# Check if at least one argument (directory) is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

new_name_prefix="$1" 

# Check if the provided directory exists
if [ ! -d "$1" ]; then
    echo "Error: Directory '$1' not found."
    exit 1
fi

# Move to the specified directory
cd "$1"

# Loop through all files in the directory
for file in *; do
    # Check if the file is a regular file (not a directory)
    if [ -f "$file" ]; then
        # Extract file name without extension
        filename="${file%.*}"
        # Extract first extension
        first_extension="${filename##*.}"
        # Check if the filename contains a dot (indicating a double extension)
        if [ "$first_extension" != "$filename" ]; then
            # Double extension detected
            filename="${filename%.*}"
            second_extension="${file##*.}"
            new_filename="${new_name_prefix}.${first_extension}.${second_extension}"
        else
            # Single extension
            extension="${file##*.}"
            new_filename="${new_name_prefix}.${extension}"
        fi
        # Rename the file
        mv "$file" "$new_filename"
        echo "Renamed '$file' to '$new_filename'"
    fi
done
